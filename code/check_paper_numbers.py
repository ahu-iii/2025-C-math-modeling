"""论文数字核对：output/*.csv 是唯一数据来源，论文初稿.md 里引用的每个统计量都必须能在
其中找到对应值（按论文写出的小数位四舍五入后比较）。不建新的"claims" CSV——直接读现成产出。

只识别口径明确的模式（OR=、IRR=、p=、p_FDR=、χ²=、95%CI 区间），宁可漏报也不臆测。
"""

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output"
DRAFT = ROOT / "docs" / "论文初稿.md"


def load_numeric_pool(out_dir: Path) -> list[float]:
    """扫 output/ 下所有 csv 的每个单元格，能转 float 的都收进池子。"""
    pool = []
    for csv_path in sorted(out_dir.rglob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                for cell in row:
                    try:
                        pool.append(float(cell))
                    except ValueError:
                        continue
    return pool


def in_pool(cited: str, pool: list[float]) -> bool:
    """cited 是论文里写的数字文本。按其（尾数部分）小数位数四舍五入来比较。

    普通形式如 "2.07"：直接四舍五入池子里的值比较。
    科学计数法形式如 "8.38e-32"（由 LaTeX \\times10^{} 或 e 记法归一化而来）：
    要求尾数四舍五入相同 *且* 指数相同——不能拿 1e-32 和 1e-29 这种都"约等于 0"
    的量直接做容差比较。
    """
    m = re.fullmatch(r"(\d+\.\d+)e(-?\d+)", cited)
    if m:
        mantissa_str, exp_str = m.groups()
        decimals = len(mantissa_str.split(".")[1])
        target_mantissa = round(float(mantissa_str), decimals)
        target_exp = int(exp_str)
        for v in pool:
            if v == 0:
                continue
            v_exp = math.floor(math.log10(abs(v)))
            v_mantissa = v / (10**v_exp)
            if v_exp == target_exp and round(v_mantissa, decimals) == target_mantissa:
                return True
        return False

    decimals = len(cited.split(".")[1]) if "." in cited else 0
    target = round(float(cited), decimals)
    return any(round(v, decimals) == target for v in pool)


# (标签, 正则, 捕获组里含数字的组号列表)
PATTERNS = [
    ("OR=", re.compile(r"OR=(\d+\.\d+)"), [1]),
    ("OR **", re.compile(r"OR\s+\*\*(\d+\.\d+)\*\*"), [1]),
    ("IRR=", re.compile(r"IRR=(\d+\.\d+)"), [1]),
    ("p_FDR=", re.compile(r"p_FDR=(\d+\.\d+)"), [1]),
    # p= / 校正后 p= / FDR p= 统一按 "紧邻 p=" 抓，前面不能是字母数字下划线（避免 p_FDR= 被重复算，
    # 也避免误配 "exp=" 这类含字母 p 的标识符）。尾数后可选跟科学计数法后缀
    # （LaTeX 的 \times10^{-32} 或代码注释里的 e-29 记法），两种都识别。
    (
        "p=",
        re.compile(
            r"(?<![A-Za-z0-9_])p\s*=\s*(\d+\.\d+)"
            r"(?:\\times10\^\{(-?\d+)\}|[eE](-?\d+))?"
        ),
        [1],
    ),
    ("χ²=", re.compile(r"χ[²2]\s*=\s*(\d+\.\d+)"), [1]),
    ("95%CI", re.compile(r"(?:95%\s?CI|(?<![A-Za-z])CI)\s*(\d+\.\d+)[–-](\d+\.\d+)"), [1, 2]),
]


def scan_draft(text: str):
    """返回 [(行号, 原始匹配文本, [引用的数字字符串...])]"""
    lines_start = [0]
    for m in re.finditer("\n", text):
        lines_start.append(m.end())

    def line_of(pos):
        lo, hi = 0, len(lines_start) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if lines_start[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    hits = []
    for label, pat, groups in pat_iter():
        for m in pat.finditer(text):
            nums = [m.group(g) for g in groups]
            if label == "p=":
                # 尾数后若跟了科学计数法后缀（\times10^{} 或 e），归一化成 "尾数e指数"
                exponent = m.group(2) or m.group(3)
                if exponent is not None:
                    nums = [f"{nums[0]}e{exponent}"]
            hits.append((line_of(m.start()), m.group(0), label, nums))
    hits.sort(key=lambda h: h[0])
    return hits


def pat_iter():
    return PATTERNS


def main():
    draft_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DRAFT
    pool = load_numeric_pool(OUT_DIR)
    text = draft_path.read_text(encoding="utf-8")
    hits = scan_draft(text)

    matched, unmatched = [], []
    for line_no, raw, label, nums in hits:
        ok = all(in_pool(n, pool) for n in nums)
        (matched if ok else unmatched).append((line_no, raw, label, nums))

    print(f"共扫到 {len(hits)} 处引用，匹配 {len(matched)}，未匹配 {len(unmatched)}")
    if unmatched:
        print("\n未匹配项：")
        for line_no, raw, label, nums in unmatched:
            print(f"  第{line_no}行 [{label}] {raw!r}  数字={nums}")

    sys.exit(1 if unmatched else 0)


if __name__ == "__main__":
    main()
