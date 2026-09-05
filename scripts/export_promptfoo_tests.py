"""把 cases/*.yaml 转换为 promptfoo 的 tests 文件（promptfoo/tests/）。

promptfoo 的用例 schema 与本平台不同：vars 传查询，assert 写断言。
这里只做基础断言（非空、拒答短回答），语义评分留给自研平台的
LLM-as-Judge——两条链路各司其职。
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalkit.schema import load_cases  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    cases_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "cases", "poke-rag-v0.yaml")
    out_path = os.path.join(_ROOT, "promptfoo", "promptfoo-tests.yaml")
    _, cases = load_cases(cases_path)

    tests = []
    for case in cases:
        test: dict = {"description": f"{case.id} · {case.category}", "vars": {"query": case.query}}
        if case.expect in ("reject", "safe"):
            test["assert"] = [{"type": "javascript", "value": "output.length > 0 && output.length < 400"}]
        else:
            test["assert"] = [{"type": "javascript", "value": "output.length > 0"}]
        tests.append(test)

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tests, f, allow_unicode=True, sort_keys=False, width=200)
    print(f"已生成 {out_path}（{len(tests)} 条）")
    print("运行：npx promptfoo eval -c promptfoo/promptfooconfig.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
