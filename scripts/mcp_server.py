"""MCP 服务入口（stdio）。配置进 MCP 客户端：
{"command": "<python>", "args": ["本文件路径"], "env": {"LLM_EVAL_ROOT": "<仓库路径>"}}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.server import mcp  # noqa: E402

if __name__ == "__main__":
    mcp.run(transport="stdio")
