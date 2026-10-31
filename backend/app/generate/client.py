"""模型客户端：DeepSeek（OpenAI 兼容接口）。

只实现 M0 需要的 ``/chat/completions``，不引入 openai SDK——少一个依赖，
也方便以后换供应商时只改 ``base_url``（任何 OpenAI 兼容端点都能直接用）。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

import httpx

from app.generate.cache import ResponseCache

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """模型调用失败。带上状态码和响应片段，便于定位是配额、鉴权还是超时。"""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ToolCall:
    """模型要求执行的一次工具调用。

    ``arguments`` **保留原样字符串**，不在客户端解析：M4 要统计"arguments 是不是合法
    JSON"这个比例（闸门 C），提前解析会把非法 JSON 变成异常，指标就丢了。
    需要字典时用 ``parsed_arguments()``，它把 JSON 错误原样抛出来供调用方记账。
    """

    id: str
    name: str
    arguments: str = ""

    def parsed_arguments(self) -> dict[str, Any]:
        """解析成 dict。非法 JSON / 非对象都抛 ``ValueError``，由调用方决定怎么降级。"""
        raw = (self.arguments or "").strip()
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments 不是合法 JSON：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"arguments 应为 JSON 对象，实际是 {type(value).__name__}")
        return value

    @property
    def is_valid_json(self) -> bool:
        try:
            self.parsed_arguments()
        except ValueError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            arguments=str(data.get("arguments") or ""),
        )


@dataclass
class ChatResult:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: 模型这一轮要求的工具调用；为空表示它给了正文（或拒答）
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: ``stop`` / ``tool_calls`` / ``length``……原样保留，循环靠它判断"是不是还要继续"
    finish_reason: str = "stop"
    #: 流式调用被取消（客户端断开/用户停止）：此时 text 只是半截，**不得**当答案交付
    cancelled: bool = False

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """缓存用的可 JSON 序列化视图（``tool_calls`` 要展开成 dict）。"""
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "cancelled": self.cancelled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResult":
        return cls(
            text=data.get("text") or "",
            model=data.get("model") or "",
            latency_ms=data.get("latency_ms") or 0.0,
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            tool_calls=[ToolCall.from_dict(item) for item in data.get("tool_calls") or []],
            finish_reason=data.get("finish_reason") or "stop",
            cancelled=bool(data.get("cancelled", False)),
        )


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_s: float = 120.0,
        max_retries: int = 2,
        cache_path: Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        #: 开启后同一请求直接返回缓存结果，使重复运行逐字一致
        self.cache = ResponseCache(cache_path) if cache_path else None

    @property
    def endpoint(self) -> str:
        # DeepSeek 同时支持 /chat/completions 与 /v1/chat/completions；
        # 已经带 /v1 的 base_url 不再重复拼接。
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        cancel_event: Event | None = None,
    ) -> ChatResult:
        """一次 ``/chat/completions``。

        ``tools`` / ``tool_choice`` 只在非空时才写进请求体——**不写**和写 ``null`` 在
        缓存键上是两回事，会把老缓存全部失效。所以默认路径的 payload 与 M3 逐字一致。

        ``tool_choice`` 接 OpenAI 兼容的三种形态：``"auto"`` / ``"none"`` /
        ``"required"``，或 ``{"type": "function", "function": {"name": ...}}``。
        """
        if not self.api_key:
            raise LLMError("未配置 DEEPSEEK_API_KEY（可写入 .env）")

        # 传入 cancel_event 时走"可取消的研究轮"：HTTP 层改成 streaming，
        # 才能在两帧之间发现取消；不传时保持原来的非流式路径（CLI/评测逐字不变）。
        if cancel_event is not None:
            return self._chat_cancellable(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                cancel_event=cancel_event,
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        cache_key = None
        if self.cache is not None:
            # 键 = (模型, 整个 payload)。payload 里现在多了 tools/tool_choice，
            # 于是"同问题、不同工具集"天然是两个缓存条目，语义仍是"payload 变则键变"。
            cache_key = self.cache.make_key(self.model, payload)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return ChatResult.from_dict(cached)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(self.endpoint, json=payload, headers=headers)
                if resp.status_code >= 400:
                    body = resp.text[:500]
                    # 4xx 里只有 429 值得重试，其余直接失败，避免白白烧时间
                    if resp.status_code >= 500 or resp.status_code == 429:
                        last_error = LLMError(
                            f"模型接口返回 {resp.status_code}", status=resp.status_code, body=body
                        )
                        self._sleep(attempt)
                        continue
                    raise LLMError(
                        f"模型接口返回 {resp.status_code}: {body}",
                        status=resp.status_code,
                        body=body,
                    )
                data = resp.json()
                latency = (time.perf_counter() - started) * 1000
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError(f"响应缺少 choices: {str(data)[:300]}")
                choice = choices[0]
                message = choice.get("message") or {}
                usage = data.get("usage") or {}
                result = ChatResult(
                    text=(message.get("content") or "").strip(),
                    model=data.get("model") or self.model,
                    latency_ms=round(latency, 1),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    tool_calls=_parse_tool_calls(message.get("tool_calls")),
                    finish_reason=str(choice.get("finish_reason") or "stop"),
                )
                if self.cache is not None and cache_key is not None:
                    self.cache.put(cache_key, result.to_dict())
                return result
            except httpx.HTTPError as exc:
                last_error = LLMError(f"网络错误：{type(exc).__name__}: {exc}")
                self._sleep(attempt)

        raise LLMError(f"模型调用失败（已重试 {self.max_retries} 次）：{last_error}")

    def _chat_cancellable(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        cancel_event: Event,
    ) -> ChatResult:
        """研究轮的"可取消"实现：HTTP 也用 streaming，在两帧之间检查取消。

        非流式的 ``httpx.post`` 一旦发出就无法中断；研究轮又占 token 大头，
        所以服务端停止必须把研究轮也改成流式读取。tool_calls 的 arguments 是
        分片到达的，这里按 ``index`` 累积后再还原成原有的 ``ToolCall``。
        """
        if cancel_event.is_set():
            return ChatResult(
                text="", model=self.model, latency_ms=0.0, finish_reason="cancelled", cancelled=True
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            text_parts: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            usage: dict[str, Any] = {}
            finish_reason = "stop"
            model = self.model
            cancelled = False
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    with client.stream(
                        "POST", self.endpoint, json=payload, headers=headers
                    ) as resp:
                        if resp.status_code >= 400:
                            body = resp.read().decode("utf-8", errors="replace")[:500]
                            if resp.status_code >= 500 or resp.status_code == 429:
                                last_error = LLMError(
                                    f"模型接口返回 {resp.status_code}",
                                    status=resp.status_code,
                                    body=body,
                                )
                                self._sleep(attempt)
                                continue
                            raise LLMError(
                                f"模型接口返回 {resp.status_code}: {body}",
                                status=resp.status_code,
                                body=body,
                            )

                        for line in resp.iter_lines():
                            if cancel_event.is_set():
                                cancelled = True
                                break
                            if not line or line.startswith(":"):
                                continue
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            usage = chunk.get("usage") or usage
                            model = chunk.get("model") or model
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta") or {}
                            piece = delta.get("content") or ""
                            if piece:
                                text_parts.append(piece)
                            for position, item in enumerate(delta.get("tool_calls") or []):
                                index = int(item.get("index", position))
                                slot = calls.setdefault(
                                    index, {"id": "", "name": "", "arguments": ""}
                                )
                                if item.get("id"):
                                    slot["id"] = str(item["id"])
                                function = item.get("function") or {}
                                if function.get("name"):
                                    slot["name"] += str(function["name"])
                                if function.get("arguments"):
                                    slot["arguments"] += str(function["arguments"])
                            if choice.get("finish_reason"):
                                finish_reason = str(choice["finish_reason"])

                return ChatResult(
                    text="".join(text_parts).strip(),
                    model=model,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    tool_calls=_stream_tool_calls(calls),
                    finish_reason="cancelled" if cancelled else finish_reason,
                    cancelled=cancelled,
                )
            except httpx.HTTPError as exc:
                if text_parts or calls:
                    raise LLMError(f"研究轮流式中断：{type(exc).__name__}: {exc}") from exc
                last_error = LLMError(f"网络错误：{type(exc).__name__}: {exc}")
                self._sleep(attempt)

        raise LLMError(f"模型调用失败（已重试 {self.max_retries} 次）：{last_error}")

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        on_token: Callable[[str], None] | None = None,
        cancel_event: Event | None = None,
    ) -> ChatResult:
        """流式 ``/chat/completions``，只用于**最终合成**。

        设计取舍：

        - 研究轮仍走 ``chat``：它要返回 JSON/tool_calls，流式解析没有收益；
        - 合成轮只发**一次**请求，边收边回调，不先非流式拿全文再伪造流式；
        - ``stream_options.include_usage`` 让最后一帧带 usage，token 账能对上；
        - 已输出过 token 后网络失败**不重试**，否则前端会看到重复半截；
        - ``cancel_event`` 在每帧之间检查：客户端断开后尽快停止继续烧 token。
        """
        if not self.api_key:
            raise LLMError("未配置 DEEPSEEK_API_KEY（可写入 .env）")
        if cancel_event is not None and cancel_event.is_set():
            return ChatResult(
                text="",
                model=self.model,
                latency_ms=0.0,
                finish_reason="cancelled",
                cancelled=True,
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            pieces: list[str] = []
            usage: dict[str, Any] = {}
            finish_reason = "stop"
            model = self.model
            cancelled = False
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    with client.stream(
                        "POST", self.endpoint, json=payload, headers=headers
                    ) as resp:
                        if resp.status_code >= 400:
                            body = resp.read().decode("utf-8", errors="replace")[:500]
                            if resp.status_code >= 500 or resp.status_code == 429:
                                last_error = LLMError(
                                    f"模型接口返回 {resp.status_code}",
                                    status=resp.status_code,
                                    body=body,
                                )
                                self._sleep(attempt)
                                continue
                            raise LLMError(
                                f"模型接口返回 {resp.status_code}: {body}",
                                status=resp.status_code,
                                body=body,
                            )

                        for line in resp.iter_lines():
                            if cancel_event is not None and cancel_event.is_set():
                                cancelled = True
                                break
                            if not line or line.startswith(":"):
                                continue
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                # 网关偶尔插入非 JSON 的保活行；忽略，不打断已收内容。
                                continue
                            usage = chunk.get("usage") or usage
                            model = chunk.get("model") or model
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta") or {}
                            piece = delta.get("content") or ""
                            if piece:
                                pieces.append(piece)
                                if on_token is not None:
                                    try:
                                        on_token(piece)
                                    except Exception:
                                        # 展示层异常不能打断流；与 Trace.on_step 同口径。
                                        pass
                            if choice.get("finish_reason"):
                                finish_reason = str(choice["finish_reason"])

                return ChatResult(
                    text="".join(pieces).strip(),
                    model=model,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason="cancelled" if cancelled else finish_reason,
                    cancelled=cancelled,
                )
            except httpx.HTTPError as exc:
                if pieces:
                    # 已经吐给前端半截，重试会造成重复；交给 _safe_chat 降级。
                    raise LLMError(f"流式中断：{type(exc).__name__}: {exc}") from exc
                last_error = LLMError(f"网络错误：{type(exc).__name__}: {exc}")
                self._sleep(attempt)

        raise LLMError(f"模型流式调用失败（已重试 {self.max_retries} 次）：{last_error}")

    @staticmethod
    def _sleep(attempt: int) -> None:
        time.sleep(min(2**attempt, 8) * 0.5)


def _stream_tool_calls(calls: dict[int, dict[str, str]]) -> list[ToolCall]:
    """把流式分片累积的 tool_calls 还原成与非流式一致的形状。"""
    out: list[ToolCall] = []
    for index in sorted(calls):
        slot = calls[index]
        out.append(
            ToolCall(
                id=slot.get("id") or f"call_{index}",
                name=slot.get("name") or "",
                arguments=slot.get("arguments") or "",
            )
        )
    return out


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    """把响应里的 ``tool_calls`` 摊平成 :class:`ToolCall`。

    容忍三种真实形态：缺失 / 空列表 / 不是列表（模型或网关抽风）。
    解析失败**不抛错**——一次无法识别的工具调用应该由循环记成"这一轮没发出工具调用"，
    而不是把整次问答打成异常（M4 兜底的要求：任一步失败都要能降级交答案）。
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=str(function.get("name") or ""),
                arguments=str(function.get("arguments") or ""),
            )
        )
    return calls


def assistant_message(result: ChatResult) -> dict[str, Any]:
    """把一轮 ``ChatResult`` 还原成 messages 里的 assistant 消息。

    回灌工具结果时必须原样带上 ``tool_calls``（含 ``id``），否则模型无法把
    ``role=tool`` 的 ``tool_call_id`` 对上是回给哪一次调用的。
    """
    message: dict[str, Any] = {"role": "assistant", "content": result.text or ""}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in result.tool_calls
        ]
    return message


def tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    """一条 ``role=tool`` 回灌消息。"""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def tool_specs(specs: list[dict]) -> list[dict]:
    """OpenAI 兼容的 ``tools`` 数组（只做形状校验，避免循环里手写失误）。"""
    out: list[dict] = []
    for spec in specs:
        if "type" not in spec or "function" not in spec:
            raise ValueError(f"工具定义缺少 type/function：{spec!r}")
        out.append(spec)
    return out
