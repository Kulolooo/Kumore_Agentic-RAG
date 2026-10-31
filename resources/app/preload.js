const { contextBridge } = require("electron");

// 只暴露只读的桌面标记，不向渲染进程开放 Node/文件系统。
contextBridge.exposeInMainWorld("kumoreDesktop", {
  isDesktop: true,
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  },
});
