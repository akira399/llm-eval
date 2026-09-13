"""MCP 服务入口。

stdio（本地，默认）：python scripts/mcp_server.py
HTTP（远程）　　：python scripts/mcp_server.py --transport http --port 8801
MCP 客户端配置（stdio）：{"command": "<python>", "args": ["本文件"],
                          "env": {"LLM_EVAL_ROOT": "<仓库路径>"}}
远程客户端连接：url = http://<host>:<port>/mcp，请求头带 Bearer API Key。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
