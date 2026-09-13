"""用户自定义被测目标配置（Phase 2.6）：让用户把自己的 AI 应用接入评测。

配置文件：<仓库根>/targets.local.json（gitignore，用户本地的"注册表"）：

    {
      "pokerag_root": "E:/path/to/poke-rag",
      "http_targets": [
        {"target_id": "my-app", "base_url": "http://127.0.0.1:9000/invoke",
         "description": "我自己的应用"}
      ]
    }

纪律：
- target_id 必须是小写字母/数字/连字符（防止路径/注入类字符）；
- 不得覆盖内置目标 id（内置的连接细节在平台代码里，用户同名会被拒绝）；
- HTTP 目标的 base_url 校验 scheme 与云元数据黑名单（与 HttpJsonAdapter 同源）；
- 文件每次读取（小文件），用户在工作台改完立即生效，无需重启。
"""
from __future__ import annotations

import json
import os
import re
import tempfile

from evalkit.providers.http_json import HttpJsonAdapter

TARGET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")
BLOCKED_HOSTS = ("169.254.169.254", "metadata.google.internal")


def config_path() -> str:
    """targets.local.json 的位置：LLM_EVAL_ROOT 优先，否则仓库根。"""
    root = os.environ.get("LLM_EVAL_ROOT") or \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "targets.local.json")


def load_user_config() -> dict:
    path = config_path()
    if not os.path.exists(path):
        return {"pokerag_root": "", "http_targets": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"pokerag_root": "", "http_targets": []}  # 用户配置坏了不拖垮平台
    return {
        "pokerag_root": str(data.get("pokerag_root") or ""),
        "http_targets": [t for t in (data.get("http_targets") or []) if isinstance(t, dict)],
    }


def save_user_config(pokerag_root: str | None = None,
                     http_targets: list[dict] | None = None) -> dict:
    """整体保存（调用方先 load 再改再存）。返回生效后的配置。"""
    current = load_user_config()
    if pokerag_root is not None:
        current["pokerag_root"] = str(pokerag_root or "").strip()
    if http_targets is not None:
        current["http_targets"] = http_targets
    fd, tmp = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    os.replace(tmp, config_path())  # 原子写，防半截文件
    return current


def _validate_http_target(target: dict) -> str | None:
    """返回错误消息；None 表示合法。"""
    tid = str(target.get("target_id", ""))
    if not TARGET_ID_RE.match(tid):
        return f"目标 id 只能是小写字母/数字/连字符（2~32 位）：{tid!r}"
    from evalkit.registry import BUILTIN_TARGET_IDS

    if tid in BUILTIN_TARGET_IDS:
        return f"目标 id 与内置目标冲突：{tid}"
    url = str(target.get("base_url", ""))
    if not url.startswith(("http://", "https://")):
        return f"base_url 必须是 http(s) 地址：{url!r}"
    if any(h in url for h in BLOCKED_HOSTS):
        return "base_url 指向被禁止的元数据地址"
    return None


def add_http_target(target_id: str, base_url: str, description: str = "") -> dict:
    target = {"target_id": target_id.strip(), "base_url": base_url.strip(),
              "description": (description or "用户自定义 HTTP 目标").strip()}
    if err := _validate_http_target(target):
        raise ValueError(err)
    config = load_user_config()
    if any(t.get("target_id") == target["target_id"] for t in config["http_targets"]):
        raise ValueError(f"目标已存在：{target['target_id']}（请先删除或换一个 id）")
    config["http_targets"].append(target)
    save_user_config(http_targets=config["http_targets"])
    return target


def remove_http_target(target_id: str) -> dict:
    config = load_user_config()
    remaining = [t for t in config["http_targets"] if t.get("target_id") != target_id]
    if len(remaining) == len(config["http_targets"]):
        raise ValueError(f"未找到自定义目标：{target_id}")
    save_user_config(http_targets=remaining)
    return {"removed": target_id}


def pokerag_root_override() -> str:
    """用户在 targets.local.json 里配置的 poke-rag 目录（空串 = 未配置）。"""
    return load_user_config()["pokerag_root"]


def build_user_http_target(target_id: str) -> HttpJsonAdapter:
    for t in load_user_config()["http_targets"]:
        if t.get("target_id") == target_id:
            return HttpJsonAdapter(base_url=t["base_url"], target_id=target_id)
    raise KeyError(f"未找到自定义 HTTP 目标：{target_id}")
