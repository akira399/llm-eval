"""创建租户并签发 API Key（本地引导操作：直接访问 jobs.db，无需已有 Key）。

用法：
  python scripts/create_tenant.py --name "团队A"
输出 tenant_id + api_key（明文只此一次）。吊销：python scripts/create_tenant.py --revoke llev_xxx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.service import EvalService, get_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="创建租户 / 吊销 API Key")
    parser.add_argument("--name", help="租户名称")
    parser.add_argument("--revoke", help="吊销指定的 API Key（明文）")
    args = parser.parse_args()

    service = get_service()
    if args.revoke:
        print(service.revoke_api_key(args.revoke))
        return 0
    if not args.name:
        parser.error("需要 --name（或 --revoke）")
    created = service.create_tenant(args.name)
    print(f"租户已创建：{created['tenant_id']}")
    print(f"API Key：{created['api_key']}")
    print(created["warning"])
    print("用法：请求头 Authorization: Bearer <api_key>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
