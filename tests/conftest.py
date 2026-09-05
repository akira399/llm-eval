import os
import sys

# 让 evalkit 可导入（tests/ 的上一级即 llm-eval 根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
