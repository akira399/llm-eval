"""失败归因 CLI（方案 §6）：对一次运行里的失败用例做两级定位。

用法：
  ../poke-rag/.venv/Scripts/python.exe scripts/attribute.py [runs/xxx.jsonl]
不指定时用最新一份运行记录。
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.attribution import attribute_run  # noqa: E402
from evalkit.schema import load_cases  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if len(sys.argv) > 1:
        run_file = sys.argv[1]
        if not os.path.isabs(run_file):
            run_file = os.path.join(_ROOT, run_file)
    else:
        matches = sorted(glob.glob(os.path.join(_ROOT, "runs", "*.jsonl")), key=os.path.getmtime)
        if not matches:
            raise FileNotFoundError("runs/ 里没有评测记录")
        run_file = matches[-1]

    with open(run_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    # 运行记录内嵌运行当时的用例快照；归因老记录时用当前用例集补新元数据
    _, cases = load_cases(os.path.join(_ROOT, "cases", "poke-rag-v0.yaml"))
    result = attribute_run(records, cases={c.id: c for c in cases})

    failures = {cid: a for cid, a in result.items() if a["failed"]}
    print(f"===== 失败归因 · {os.path.basename(run_file)} · 失败 {len(failures)} 条（通过 {len(records) - len(failures)}）=====")
    by_layer: dict[str, list] = {}
    for cid, a in failures.items():
        by_layer.setdefault(a["layer"], []).append((cid, a))
    for layer, items in sorted(by_layer.items()):
        print(f"\n【{layer}】{len(items)} 条")
        for cid, a in items:
            print(f"  {cid} · {a['label']}")
            if a.get("detail"):
                print(f"      {a['detail'][:120]}")

    out_dir = os.path.join(_ROOT, "runs", "attributions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(run_file))[0] + ".attribution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
