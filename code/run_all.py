"""复现流水线：按顺序运行三个问题的脚本。

三个脚本互不 import，也不读取彼此的输出文件，因此顺序不是正确性要求；
按 problem1 -> problem2 -> problem3 排列只是为了输出顺序符合直觉。
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
