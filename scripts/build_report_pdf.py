"""《LLM 应用效果评测报告》PDF 构建：markdown → HTML → Edge 无头打印。

与 poke-rag 的 build_lessons_pdf.py 同一套方案（本机 Edge + 系统中文字体）。
运行：../poke-rag/.venv/Scripts/python.exe scripts/build_report_pdf.py
产物：reports/LLM应用效果评测报告.pdf（gitignore，作为简历附件的最终版式）
"""
import os
import subprocess
import sys

import markdown

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MD = os.path.join(_ROOT, "reports", "LLM应用效果评测报告.md")
OUT_PDF = os.path.join(_ROOT, "reports", "LLM应用效果评测报告.pdf")

CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
       font-size: 10.5pt; line-height: 1.8; color: #1f2328; margin: 0; }
h1 { font-size: 17pt; border-bottom: 3px solid #0969da; padding-bottom: 8px; margin: 0 0 16px; }
h2 { font-size: 13.5pt; border-left: 5px solid #0969da; padding-left: 10px;
     margin: 24px 0 10px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 18px 0 8px; }
blockquote { background: #f0f6ff; border-left: 4px solid #0969da;
             margin: 12px 0; padding: 10px 14px; color: #1c3d5a; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
      padding: 12px 14px; white-space: pre-wrap; word-break: break-all;
      font-size: 9pt; font-family: Consolas, "Microsoft YaHei", monospace; }
code { font-family: Consolas, monospace; background: #f0f1f3; padding: 1px 5px;
       border-radius: 3px; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th { background: #eef2f7; text-align: left; }
th, td { border: 1px solid #d0d7de; padding: 6px 9px; }
tr:nth-child(even) td { background: #fafbfc; }
.cover { text-align: center; padding: 120px 0 0; page-break-after: always; }
.cover h1 { font-size: 26pt; border: none; margin-bottom: 10px; }
.cover .sub { font-size: 13pt; color: #555; margin: 8px 0; }
.cover .meta { font-size: 10.5pt; color: #777; margin-top: 40px; line-height: 2; }
"""


def build_html() -> str:
    with open(SRC_MD, encoding="utf-8") as f:
        md = f.read()
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<style>{CSS}</style></head><body>
<div class="cover">
  <h1>LLM 应用效果评测报告</h1>
  <div class="sub">LLM-Eval · LLM 应用效果评测与回归平台</div>
  <div class="sub" style="font-size:11.5pt">被测对象：Poke-RAG 宝可梦对战知识库问答系统</div>
  <div class="meta">
    用例集：50 条种子 + 150 条卡片锚定扩写 + 20 条红队边界用例<br>
    评测维度：正确性 · 引用忠实度 · 格式 · 语气（LLM-as-Judge + 人工盲评校准）<br>
    生成日期：2026-09-07
  </div>
</div>
{body}</body></html>"""


def main() -> int:
    html_path = SRC_MD.replace(".md", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html())

    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge):
        edge = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    subprocess.run([
        edge, "--headless=old", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}", "file:///" + html_path.replace("\\", "/"),
    ], check=True, timeout=120)
    print(f"PDF 生成成功: {OUT_PDF} ({os.path.getsize(OUT_PDF) / 1024:.0f} KB)")
    os.remove(html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
