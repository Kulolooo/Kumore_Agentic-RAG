/**
 * Kumore desktop shell.
 *
 * 这是一层“瘦客户端”：Electron 只负责窗口、原生菜单、启动/关闭本地 FastAPI。
 * 检索、OCR、模型、语料、索引都在 exe 同目录的 backend/ 里，方便直接改代码、换模型。
 */
const { app, BrowserWindow, Menu, dialog, shell } = require("electron");
const { spawn, execFile } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");

const IS_WIN = process.platform === "win32";
const APP_NAME = "Kumore";
const FRONTEND_WAIT_MS = 90_000;

let mainWindow = null;
let backend = null;
let backendUrl = "";
let quitting = false;
let backendLog = "";

app.setName(APP_NAME);
if (IS_WIN) app.setAppUserModelId("com.kumore.desktop");

function exeDir() {
  return path.dirname(app.getPath("exe"));
}

function firstExisting(candidates) {
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      if (fs.existsSync(candidate)) return candidate;
    } catch {
      // ignore
    }
  }
  return "";
}

function backendRoot() {
  const candidates = [
    process.env.KUMORE_BACKEND_ROOT,
    // 瘦客户端：backend 永远优先找 exe 同目录，开发/发行都用同一套。
    path.join(exeDir(), "backend"),
    path.resolve(__dirname, ".."),
    path.resolve(__dirname, "..", "..", "backend"),
  ];
  const root = firstExisting(candidates);
  if (!root) {
    throw new Error(
      "找不到 backend 目录。请把 backend/ 放在 exe 同目录，或设置 KUMORE_BACKEND_ROOT。",
    );
  }
  return root;
}

function pythonExe(root) {
  return (
    firstExisting([
      process.env.KUMORE_PYTHON,
      path.join(exeDir(), "runtime", "python.exe"),
      path.join(exeDir(), "runtime", "Scripts", "python.exe"),
      path.join(root, ".venv", "Scripts", "python.exe"),
      path.join(root, ".venv", "bin", "python"),
    ]) || "python"
  );
}

function logPath() {
  const dir = path.join(app.getPath("userData"), "logs");
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, "desktop.log");
}

function log(message) {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  backendLog += line;
  try {
    fs.appendFileSync(logPath(), line, "utf8");
  } catch {
    // 日志失败不能阻止启动。
  }
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

async function fetchJson(url, options = {}, timeoutMs = 30_000) {
  const response = await fetch(url, {
    ...options,
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`${response.status} ${body.slice(0, 300)}`);
  }
  return response.json();
}

async function waitForBackend(url) {
  const deadline = Date.now() + FRONTEND_WAIT_MS;
  let lastError = "";
  while (Date.now() < deadline) {
    try {
      const health = await fetchJson(`${url}/api/health`, {}, 2_000);
      if (health && typeof health === "object") return health;
    } catch (exc) {
      lastError = String(exc);
    }
    await new Promise((resolve) => setTimeout(resolve, 350));
  }
  throw new Error(`本地后端启动超时。${lastError}`);
}

async function startBackend() {
  const root = backendRoot();
  const entry = path.join(root, "app", "api", "app.py");
  if (!fs.existsSync(entry)) {
    throw new Error(`backend 不完整：缺少 ${entry}`);
  }
  const port = await freePort();
  const python = pythonExe(root);
  const url = `http://127.0.0.1:${port}`;
  log(`backend root=${root} python=${python} port=${port}`);

  backend = spawn(
    python,
    ["-m", "app", "web", "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd: root,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONUTF8: "1",
        // 让后端知道这是桌面壳，UI 可据此后置窗口相关行为。
        KUMORE_DESKTOP: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  const pipe = (stream) => {
    stream.setEncoding("utf8");
    stream.on("data", (chunk) => {
      const text = String(chunk);
      backendLog += text;
      try {
        fs.appendFileSync(logPath(), text);
      } catch {
        // ignore
      }
    });
  };
  if (backend.stdout) pipe(backend.stdout);
  if (backend.stderr) pipe(backend.stderr);

  backend.on("error", (exc) => {
    log(`backend spawn error: ${exc}`);
    if (!quitting) {
      dialog.showErrorBox(
        "Kumore 后端启动失败",
        `无法启动 Python 后端。\n\n${exc}\n\n日志：${logPath()}`,
      );
    }
  });
  backend.on("exit", (code, signal) => {
    log(`backend exited code=${code} signal=${signal}`);
    if (!quitting) {
      dialog.showErrorBox(
        "Kumore 后端已退出",
        `后端进程退出（code=${code}）。\n\n日志：${logPath()}\n\n${backendLog.slice(-3000)}`,
      );
      app.quit();
    }
  });

  await waitForBackend(url);
  backendUrl = url;
  log(`backend ready at ${url}`);
  return url;
}

function stopBackend() {
  quitting = true;
  if (!backend) return;
  const pid = backend.pid;
  try {
    backend.kill();
  } catch {
    // ignore
  }
  if (IS_WIN && pid) {
    // uvicorn/子进程可能不跟随父进程退出；taskkill /T 清整棵树。
    execFile("taskkill", ["/PID", String(pid), "/T", "/F"], () => {});
  }
  backend = null;
}

function windowStatePath() {
  return path.join(app.getPath("userData"), "window-state.json");
}

function loadWindowState() {
  try {
    const data = JSON.parse(fs.readFileSync(windowStatePath(), "utf8"));
    if (Number.isFinite(data.width) && Number.isFinite(data.height)) {
      return { width: data.width, height: data.height, x: data.x, y: data.y };
    }
  } catch {
    // first run
  }
  return { width: 1440, height: 920 };
}

function saveWindowState() {
  if (!mainWindow) return;
  try {
    const bounds = mainWindow.getBounds();
    fs.writeFileSync(windowStatePath(), JSON.stringify(bounds), "utf8");
  } catch {
    // ignore
  }
}

async function createWindow(url) {
  const state = loadWindowState();
  mainWindow = new BrowserWindow({
    ...state,
    minWidth: 980,
    minHeight: 640,
    show: false,
    backgroundColor: "#080b09",
    titleBarStyle: "hidden",
    titleBarOverlay: IS_WIN
      ? { color: "#080b09", symbolColor: "#a8b3aa", height: 36 }
      : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("close", saveWindowState);
  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    if (/^https?:/i.test(target) && !target.startsWith(url)) {
      shell.openExternal(target);
    }
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, target) => {
    if (!target.startsWith(url)) {
      event.preventDefault();
      if (/^https?:/i.test(target)) shell.openExternal(target);
    }
  });

  await mainWindow.loadURL(url);
  // 顶部状态条在窗控按钮下留出空间；并允许拖动空白区。
  // insertCSS 必须在页面 load 之后，否则在空白 webContents 上可能一直不 resolve。
  try {
    await mainWindow.webContents.insertCSS(`
      .topbar { -webkit-app-region: drag; padding-right: 154px !important; }
      .topbar button, .topbar .pill, .topbar .topbar-meta { -webkit-app-region: no-drag; }
      @media (max-width: 980px) { .topbar { padding-right: 150px !important; } }
    `);
  } catch (exc) {
    log(`insertCSS failed: ${exc}`);
  }
  mainWindow.show();
}

async function openKnowledgeBaseFolder() {
  if (!mainWindow || !backendUrl) return;
  const picked = await dialog.showOpenDialog(mainWindow, {
    title: "选择知识库目录",
    properties: ["openDirectory"],
  });
  if (picked.canceled || !picked.filePaths[0]) return;
  const folder = picked.filePaths[0];
  const name = path.basename(folder) || "知识库";
  mainWindow.setProgressBar(0.5);
  try {
    const kb = await fetchJson(
      `${backendUrl}/api/kbs`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, corpus_root: folder }),
      },
      120_000,
    );
    // 新建后必须重建；否则 UI 里是空索引。
    await fetchJson(
      `${backendUrl}/api/kbs/${encodeURIComponent(kb.id)}/rebuild`,
      { method: "POST" },
      30 * 60_000,
    );
    await mainWindow.loadURL(backendUrl);
  } catch (exc) {
    dialog.showErrorBox("导入知识库失败", `${exc}`);
  } finally {
    mainWindow.setProgressBar(-1);
  }
}

function buildMenu() {
  return Menu.buildFromTemplate([
    {
      label: "文件",
      submenu: [
        {
          label: "打开知识库文件夹…",
          accelerator: "CmdOrCtrl+O",
          click: () => void openKnowledgeBaseFolder(),
        },
        {
          label: "重新加载界面",
          accelerator: "CmdOrCtrl+R",
          click: () => mainWindow?.reload(),
        },
        { type: "separator" },
        { role: "quit", label: "退出" },
      ],
    },
    { role: "editMenu", label: "编辑" },
    {
      label: "视图",
      submenu: [
        { role: "toggleDevTools", label: "开发者工具" },
        { role: "resetZoom", label: "重置缩放" },
        { role: "zoomIn", label: "放大" },
        { role: "zoomOut", label: "缩小" },
      ],
    },
    {
      label: "帮助",
      submenu: [
        {
          label: "打开日志目录",
          click: () => shell.openPath(path.join(app.getPath("userData"), "logs")),
        },
        {
          label: "关于 Kumore",
          click: () =>
            dialog.showMessageBox({
              type: "info",
              title: "Kumore",
              message: `Kumore ${app.getVersion()}`,
              detail: `本地知识库客户端\n后端：${backendRoot()}\n日志：${logPath()}`,
            }),
        },
      ],
    },
  ]);
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    Menu.setApplicationMenu(buildMenu());
    try {
      const url = await startBackend();
      await createWindow(url);
    } catch (exc) {
      log(`startup failed: ${exc}`);
      dialog.showErrorBox(
        "Kumore 启动失败",
        `${exc}\n\n日志：${logPath()}\n\n${backendLog.slice(-3000)}`,
      );
      stopBackend();
      app.quit();
    }
  });

  app.on("before-quit", stopBackend);
  app.on("window-all-closed", () => {
    app.quit();
  });
}
