"""工作台启动器：自动挑选空闲端口并打开浏览器（双击 bat 的实际入口）。

为什么需要它：Streamlit 固定端口被占用时会直接报错退出、窗口一闪而过
（用户实测踩坑——上一实例还开着时再次双击就"什么都没发生"）。
启动器从 8501 起顺延找空闲端口，保证总能起来。
"""
import os
import socket
import sys
import threading
import webbrowser

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSOLE = os.path.join(_ROOT, "scripts", "console.py")


def free_port(start: int = 8501, tries: int = 20) -> int:
    """返回第一个未被占用的端口；全被占用则报错退出。"""
    for port in range(start, start + tries):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:  # 连不上 = 空闲
                return port
    raise SystemExit(f"端口 {start}~{start + tries - 1} 全部被占用，"
                     "请关闭旧的工作台窗口后重试。")


def main() -> int:
    port = free_port()
    url = f"http://localhost:{port}"
    print("=" * 46)
    print("  LLM-Eval 工作台启动中")
    print(f"  地址：{url}")
    print("  关闭本窗口即可停止服务；浏览器没自动打开时，手动访问上面的地址。")
    print("=" * 46, flush=True)
    threading.Timer(2.5, webbrowser.open, [url]).start()
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", CONSOLE,
                f"--server.port={port}", "--server.headless=true"]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
