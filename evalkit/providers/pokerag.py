"""被测应用适配器：Poke-RAG。

两种模式（方案 §2 设计决策 1）：
- inprocess：直接驱动 RAGEngine().answer()（默认，无需起服务）；
- http：POST /api/chat 解析 SSE 事件（适配"被测系统只能通过网络访问"的场景）。

「版本 = 配置快照」（方案 §2 设计决策 2）：overrides 会与 poke-rag 的
config.local.json 合并后写入临时配置文件，并把 poke-rag config 模块的
CONFIG_PATH 指过去——不改动用户的本地配置，运行元数据记录生效配置，
保证回归对比可复现。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

from evalkit.config import poke_rag_root
from evalkit.schema import TargetResult


def _deep_merge(base: dict, patch: dict) -> dict:
    merged = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class PokeRagProvider:
    def __init__(
        self,
        root: str | None = None,
        overrides: dict | None = None,
        mode: str = "inprocess",
        base_url: str = "http://127.0.0.1:8765",
    ):
        if mode not in ("inprocess", "http"):
            raise ValueError(f"mode 必须是 inprocess|http，得到 {mode!r}")
        self.root = os.path.abspath(root or poke_rag_root())
        self.overrides = overrides or {}
        self.mode = mode
        self.base_url = base_url
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"找不到被测系统目录：{self.root}（可用 --poke-rag-root 或环境变量 POKE_RAG_ROOT 指定）"
            )
        self._tmp_config: str | None = None
        if self.mode == "inprocess":
            self._setup_inprocess()

    # ---- inprocess ----

    def _setup_inprocess(self) -> None:
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        import src.config as pr_config  # poke-rag 的配置模块（与 evalkit 无命名冲突）

        self._pr_config = pr_config
        if self.overrides:
            merged = _deep_merge(pr_config.load(), self.overrides)
            fd, self._tmp_config = tempfile.mkstemp(suffix=".json", prefix="llmeval-")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            pr_config.CONFIG_PATH = self._tmp_config

    def _answer_inprocess(self, query: str) -> TargetResult:
        res = TargetResult()
        try:
            # RAGEngine 构造廉价（只读配置），每次新建避免跨用例状态；
            # 延迟 import 使本模块可被纯配置类测试轻量加载。
            from src.generation.rag import RAGEngine

            for event in RAGEngine().answer(query):
                kind = event.get("type")
                if kind == "citations":
                    res.citations = event.get("mapping", {})
                    res.cited_cards = [c for c in event.get("cards", []) if c]
                elif kind == "delta":
                    res.answer_text += event.get("text", "")
                elif kind == "reject":
                    res.rejected = True
                elif kind == "done":
                    res.citations_ok = bool(event.get("citations_ok", True))
        except Exception as exc:  # 单条用例失败不能拖垮整个运行
            res.error = f"{type(exc).__name__}: {exc}"
        return res

    # ---- http ----

    def _answer_http(self, query: str) -> TargetResult:
        import httpx

        res = TargetResult()
        try:
            with httpx.Client(timeout=120) as client:
                with client.stream(
                    "POST", f"{self.base_url}/api/chat", json={"query": query}
                ) as resp:
                    resp.raise_for_status()
                    buf = ""
                    for chunk in resp.iter_text():
                        buf += chunk
                        while "\n\n" in buf:
                            raw, buf = buf.split("\n\n", 1)
                            line = raw.strip()
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[len("data: "):])
                            kind = event.get("type")
                            if kind == "citations":
                                res.citations = event.get("mapping", {})
                                res.cited_cards = [c for c in event.get("cards", []) if c]
                            elif kind == "delta":
                                res.answer_text += event.get("text", "")
                            elif kind == "reject":
                                res.rejected = True
                            elif kind == "done":
                                res.citations_ok = bool(event.get("citations_ok", True))
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
        return res

    # ---- 对外 ----

    def answer(self, query: str) -> TargetResult:
        t0 = time.perf_counter()
        if self.mode == "http":
            res = self._answer_http(query)
        else:
            res = self._answer_inprocess(query)
        res.latency_ms = int((time.perf_counter() - t0) * 1000)
        return res

    def effective_config(self) -> dict:
        """运行元数据：这次评测的被测系统到底跑在什么配置上（可复现的依据）。

        api_key 只保留脱敏形式——元数据会进 summary.json，密钥不能落盘明文。
        """
        if self.mode != "inprocess":
            return {"mode": self.mode, "base_url": self.base_url}
        cfg = self._pr_config.load()
        key = cfg.get("llm", {}).get("api_key") or ""
        if key:
            cfg["llm"]["api_key"] = f"{key[:3]}****{key[-4:]}" if len(key) > 6 else "****"
        return cfg
