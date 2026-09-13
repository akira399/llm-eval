"""工作台启动器测试：端口顺延（8501 被占时自动换下一个）。"""
import socket

from scripts.launch_console import free_port


def test_free_port_skips_occupied():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        occupied = s.getsockname()[1]
        picked = free_port(start=occupied, tries=5)
        assert picked > occupied
