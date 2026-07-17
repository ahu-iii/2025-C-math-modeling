"""复现流水线：按顺序运行三个问题的脚本。

注意：problem3 会 import problem2 作为模块，因此顺序必须是
problem1 -> problem2 -> problem3；这里用一个显式的有序列表即可，
不需要额外的依赖图。
"""
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).parent
SCRIPTS = ["problem1_baseline.py", "problem2_factors.py", "problem3_efficacy.py"]

for script in SCRIPTS:
    print(f"==> running {script}", flush=True)
    result = subprocess.run([sys.executable, str(CODE_DIR / script)])
    if result.returncode != 0:
        print(f"==> {script} failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)

print("==> all scripts completed successfully", flush=True)
