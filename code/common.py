"""三问共用的口径：路径、附件1 列名、分组/药物标签、队列构造、中文字体。

单一权威定义处，避免 problem1/2/3 各自实现同一口径而分叉（如缓解/复发/D_t 的构造）。
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
F1 = ROOT / "附件1：两个院临床受试者及抑郁症的基本数据.xlsx"
F2 = ROOT / "附件2：两个医院随访的抗抑郁药使用后主诉情况.xlsx"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# 论文插图出 OUT_DIR，诊断图（自查用、不入论文）出 DIAG_DIR。
# 准入检验见 docs/adr/0011-出图规范论文插图与诊断图分离.md。
DIAG_DIR = OUT_DIR / "diagnostics"
DIAG_DIR.mkdir(exist_ok=True)

# 分组/药物标签的唯一权威定义；GROUP_LABELS 由 GROUPS 派生（组N-药物X 的形式）。
GROUPS = {1: "药物C(对照)", 2: "药物A", 3: "药物B"}
GROUP_LABELS = {g: f"组{g}-{name}" for g, name in GROUPS.items()}

ALPHA = 0.05

# 附件1 列名
COL_NAMES = [
    "序号", "组别", "年龄",
    "未婚", "已婚", "离异", "丧偶",
    "无用药史", "使用过抗抑郁药", "其它用药史",
    "轻度", "中度", "重度",
]

# 附件2 相关常量
TIMES = [1, 3, 6, 12]
SYMPTOMS = ["失访", "有自杀倾向", "副作用导致停药", "失眠", "脱发", "激素水平异常", "嗜睡", "便秘"]
# 7 类实质性不适；失访是"缺失/失访"而非症状，D_t 不计入(ADR-0001)、B_t/症状负担也不计入(ADR-0005)。
# 已核实：附件2 现成的"是否出现不适症状"栏 == 这 7 类之和（4 时点 0 误差），佐证 失访 应排除。
SUBST = SYMPTOMS[1:]


# --------------------------------------------------------------------------
# 中文字体（macOS 系统自带，避免图中出现方框）
# --------------------------------------------------------------------------
def setup_cjk_font():
    candidates = [
        "/System/Library/Fonts/Supplemental/Heiti SC.ttc",  # 可能不存在, 保留兜底
        "/System/Library/Fonts/Supplemental/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            matplotlib.font_manager.fontManager.addfont(str(p))
            name = matplotlib.font_manager.FontProperties(fname=str(p)).get_name()
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    warnings.warn("未找到中文字体，图中文字可能显示为方框。")


setup_cjk_font()


# --------------------------------------------------------------------------
# 数据加载与整理
# --------------------------------------------------------------------------
def onehot_to_category(df, cols, out_name):
    """一组 one-hot 列坍缩为单个分类变量；全空记为'未知'（缺失/失访）。

    已知边界情形：附件1 有 3/3149 受试者同一组内命中>1 个 one-hot（录入笔误），此时
    idxmax 取 cols 中靠前者（如 已婚+丧偶→已婚）；样本量极小、不影响结论。"""
    sub = df[cols]
    cat = sub.fillna(0).idxmax(axis=1).where(sub.eq(1).any(axis=1), other="未知")
    cat.name = out_name
    return cat


def load_baseline():
    frames = []
    for sheet, hosp in [("一院临床受试者及抑郁症的基本数据", "一院"),
                        ("二院临床受试者及抑郁症的基本数据", "二院")]:
        d = pd.read_excel(F1, sheet_name=sheet, header=None, skiprows=2,
                          usecols=range(len(COL_NAMES)), names=COL_NAMES).dropna(how="all")
        d[COL_NAMES[3:]] = d[COL_NAMES[3:]].apply(pd.to_numeric, errors="coerce")
        d["医院"] = hosp
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["组别"] = df["组别"].astype(int)
    df["婚姻状况"] = onehot_to_category(df, ["未婚", "已婚", "离异", "丧偶"], "婚姻状况")
    df["既往用药"] = onehot_to_category(df, ["无用药史", "使用过抗抑郁药", "其它用药史"], "既往用药")
    df["抑郁程度"] = onehot_to_category(df, ["轻度", "中度", "重度"], "抑郁程度")
    df["年龄"] = pd.to_numeric(df["年龄"], errors="coerce")
    return df[["医院", "序号", "组别", "年龄", "婚姻状况", "既往用药", "抑郁程度"]]


def load_followup():
    """读取附件2 并按 (医院,序号,组别) 去重：同一受试者的多行症状标记取并集(union)。"""
    sympt_cols = [f"{s}@{t}" for s in SYMPTOMS for t in TIMES]
    names = ["序号", "组别"] + [f"{s}@{t}" for s in SYMPTOMS + ["是否出现不适症状"] for t in TIMES]
    frames = []
    for sheet, hosp in [("一院随访的抗抑郁药物使用后主诉情况", "一院"),
                        ("二院随访的抗抑郁药物使用后主诉情况", "二院")]:
        d = pd.read_excel(F2, sheet_name=sheet, header=None, skiprows=3,
                          names=names).dropna(how="all")
        d["医院"] = hosp
        d["组别"] = d["组别"].astype(int)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw[sympt_cols] = raw[sympt_cols].apply(pd.to_numeric, errors="coerce")
    n_raw = len(raw)
    fu = raw.groupby(["医院", "序号", "组别"], as_index=False)[sympt_cols].max()
    return fu, n_raw, n_raw - len(fu)


def build_cohort():
    """附件1 左连接附件2 → 全队列；缺席附件2 者所有症状记为 0（全程无不适）。构造结局。"""
    base = load_baseline()
    fu, n_raw, n_dup = load_followup()
    df = base.merge(fu, on=["医院", "序号", "组别"], how="left")

    for t in TIMES:
        df[f"B{t}"] = df[[f"{s}@{t}" for s in SUBST]].fillna(0).gt(0).sum(axis=1)
        df[f"D{t}"] = (df[f"B{t}"] > 0).astype(int)

    df["非缓解"] = df["D12"]                                    # D12=1 → 未缓解
    df["任何主诉"] = df[[f"D{t}" for t in TIMES]].max(axis=1)
    df["总症状负担"] = df[[f"B{t}" for t in TIMES]].sum(axis=1)
    # 曾缓解：宽松 D3=0 或 D6=0（ADR-0004 定案）。与缓解同为"单时点无不适"判据，只是看更早的窗口；
    # 严格版（仅 D6=0）会对"曾缓解"用比"缓解"本身更严的尺子，无依据。
    df["曾缓解"] = ((df["D3"] == 0) | (df["D6"] == 0)).astype(int)
    df["复发"] = (((df["D3"] == 0) | (df["D6"] == 0)) & (df["D12"] == 1)).astype(int)

    # 缓解：单点 D12=0（ADR-0003 定案）。不可改用两点 D6=D12=0 —— 那恰是「豁免后持续不适」
    # 的补集，两者会在综合层里变成同一根轴的正反两面（见 problem3_efficacy 模块 docstring）。
    df["缓解"] = (df["D12"] == 0).astype(int)
    # 适应期（ADR-0002）：前期(1或3月)有不适，但后期(6且12月)均无 → 视为已适应，前三月不计入药物质量问题。
    df["适应期"] = (((df["D1"] == 1) | (df["D3"] == 1)) & (df["D6"] == 0) & (df["D12"] == 0)).astype(int)
    # 豁免后持续不适：把适应期个体豁免掉之后，仍构成"药物质量问题"的人 —— 即后期(6或12月)仍有不适。
    df["豁免后持续不适"] = ((df["D6"] == 1) | (df["D12"] == 1)).astype(int)

    df["年龄"] = df["年龄"].fillna(df["年龄"].median())
    return df, {"附件2原始行数": n_raw, "去重删除行数": n_dup, "建模队列N": len(df)}
