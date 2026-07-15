"""
问题1：根据附件1，分析各组患者的基线特征（年龄、婚姻状况、既往抗抑郁药使用情况、
初始抑郁程度）的分布情况，比较各组之间是否存在显著差异。

输出：
  output/table1_baseline_summary.csv   - 完整基线特征表（Table 1）
  output/test_results.csv              - 假设检验结果明细（含多重比较校正）
  output/age_boxplot.png               - 年龄分组箱线图
  output/marital_status_bar.png        - 婚姻状况分组堆积柱状图
  output/prior_use_bar.png             - 既往用药情况分组堆积柱状图
  output/depression_severity_bar.png   - 初始抑郁程度分组堆积柱状图
  output/hospital_group_heatmap.png    - 医院 x 组别 分布核查
"""

import itertools
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "附件1：两个院临床受试者及抑郁症的基本数据.xlsx"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

GROUP_LABELS = {1: "组1-药物C(对照)", 2: "组2-药物A", 3: "组3-药物B"}
ALPHA = 0.05

COL_NAMES = [
    "序号", "组别", "年龄",
    "未婚", "已婚", "离异", "丧偶",
    "无用药史", "使用过抗抑郁药", "其它用药史",
    "轻度", "中度", "重度",
]

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
def load_hospital_sheet(sheet_name: str, hospital_label: str) -> pd.DataFrame:
    df = pd.read_excel(
        DATA_FILE, sheet_name=sheet_name, header=None, skiprows=2,
        usecols=range(len(COL_NAMES)), names=COL_NAMES,
    )
    df = df.dropna(how="all")
    onehot_cols = COL_NAMES[3:]  # 婚姻/用药史/抑郁程度的 one-hot 列，个别单元格含空字符串而非真 NaN
    df[onehot_cols] = df[onehot_cols].apply(pd.to_numeric, errors="coerce")
    df["医院"] = hospital_label
    return df


def onehot_to_category(df: pd.DataFrame, cols: list[str], out_name: str) -> pd.Series:
    """将一组 one-hot 列坍缩为单个分类变量；全空记为'未知'（缺失/失访）。"""
    sub = df[cols]
    has_any = sub.eq(1).any(axis=1)
    cat = sub.fillna(0).idxmax(axis=1)   # 取值为1的列名；全空时任意选一列，随后被覆盖
    cat = cat.where(has_any, other="未知")
    cat.name = out_name
    return cat


def build_dataset() -> pd.DataFrame:
    d1 = load_hospital_sheet("一院临床受试者及抑郁症的基本数据", "一院")
    d2 = load_hospital_sheet("二院临床受试者及抑郁症的基本数据", "二院")
    df = pd.concat([d1, d2], ignore_index=True)

    df["组别"] = df["组别"].astype(int)
    df["组别标签"] = df["组别"].map(GROUP_LABELS)

    df["婚姻状况"] = onehot_to_category(df, ["未婚", "已婚", "离异", "丧偶"], "婚姻状况")
    df["既往用药"] = onehot_to_category(
        df, ["无用药史", "使用过抗抑郁药", "其它用药史"], "既往用药"
    )
    df["抑郁程度"] = onehot_to_category(df, ["轻度", "中度", "重度"], "抑郁程度")

    return df[["序号", "医院", "组别", "组别标签", "年龄", "婚姻状况", "既往用药", "抑郁程度"]]


# --------------------------------------------------------------------------
# 描述性统计 (Table 1)
# --------------------------------------------------------------------------
def split_age_by_group(df: pd.DataFrame) -> list[pd.Series]:
    """按 GROUP_LABELS 顺序拆分各组年龄（已去除缺失值）。"""
    return [df.loc[df["组别"] == g, "年龄"].dropna() for g in GROUP_LABELS]


def summarize_age(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, label), s in zip(GROUP_LABELS.items(), split_age_by_group(df)):
        rows.append({
            "变量": "年龄", "组别": label, "n(非缺失)": s.shape[0],
            "缺失数": (df["组别"] == g).sum() - s.shape[0],
            "均值±SD": f"{s.mean():.2f} ± {s.std():.2f}",
            "中位数(IQR)": f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})",
            "范围": f"{s.min():.0f}-{s.max():.0f}",
        })
    return pd.DataFrame(rows)


def summarize_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for g, label in GROUP_LABELS.items():
        sub = df.loc[df["组别"] == g, col]
        n_total = sub.shape[0]
        counts = sub.value_counts()
        for cat, n in counts.items():
            rows.append({
                "变量": col, "组别": label, "类别": cat,
                "n": n, "占比(%)": round(100 * n / n_total, 1),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 假设检验
# --------------------------------------------------------------------------
def compute_age_diagnostics(groups: list[np.ndarray]) -> dict:
    """计算年龄的正态性(Shapiro)/方差齐性(Levene)/组间差异(ANOVA、Kruskal-Wallis)检验统计量。

    返回未四舍五入的原始值：test_age() 据此四舍五入入表，
    plot_age_test_pvalues() 需要原始精度（Shapiro p 常小至 1e-11 量级，
    四舍五入后会变成 0.0 而无法在对数坐标上绘制），故两处均调用本函数而非互相复用结果。
    """
    shapiro_p = [float(stats.shapiro(g)[1]) if 3 <= len(g) <= 5000 else np.nan for g in groups]
    _, levene_p = stats.levene(*groups)
    f_stat, anova_p = stats.f_oneway(*groups)
    h_stat, kw_p = stats.kruskal(*groups)
    return {
        "shapiro_p": shapiro_p, "levene_p": levene_p,
        "f_stat": f_stat, "anova_p": anova_p,
        "h_stat": h_stat, "kw_p": kw_p,
    }


def test_age(df: pd.DataFrame) -> dict:
    groups = [s.values for s in split_age_by_group(df)]

    diag = compute_age_diagnostics(groups)
    normal_p = diag["shapiro_p"]
    levene_p = diag["levene_p"]
    f_stat, anova_p = diag["f_stat"], diag["anova_p"]
    h_stat, kw_p = diag["h_stat"], diag["kw_p"]
    normal_ok = all(p > ALPHA for p in normal_p if not np.isnan(p))
    variance_ok = levene_p > ALPHA

    recommended = "ANOVA" if (normal_ok and variance_ok) else "Kruskal-Wallis"
    main_p = anova_p if recommended == "ANOVA" else kw_p

    result = {
        "检验对象": "年龄", "检验方法": "ANOVA / Kruskal-Wallis(见推荐列)",
        "正态性(各组Shapiro p)": [round(p, 4) if not np.isnan(p) else None for p in normal_p],
        "方差齐性(Levene p)": round(levene_p, 4),
        "ANOVA F": round(f_stat, 3), "ANOVA p": round(anova_p, 4),
        "Kruskal-Wallis H": round(h_stat, 3), "Kruskal-Wallis p": round(kw_p, 4),
        "推荐方法": recommended, "推荐方法p值": round(main_p, 4),
        "显著(未校正)": main_p < ALPHA,
    }

    # 若整体显著，做事后两两比较
    posthoc_rows = []
    if main_p < ALPHA:
        if recommended == "ANOVA":
            labels = np.concatenate([[GROUP_LABELS[g]] * len(v)
                                      for g, v in zip(GROUP_LABELS, groups)])
            values = np.concatenate(groups)
            tukey = pairwise_tukeyhsd(values, labels, alpha=ALPHA)
            for row in tukey.summary().data[1:]:
                posthoc_rows.append({
                    "组1": row[0], "组2": row[1], "均值差": row[2],
                    "p(Tukey校正)": row[3], "显著": row[6],
                })
        else:
            age_by_group = dict(zip(GROUP_LABELS, groups))
            pairs = list(itertools.combinations(GROUP_LABELS.keys(), 2))
            raw_p = []
            for g1, g2 in pairs:
                _, p = stats.mannwhitneyu(
                    age_by_group[g1], age_by_group[g2], alternative="two-sided",
                )
                raw_p.append(p)
            _, adj_p, _, _ = multipletests(raw_p, alpha=ALPHA, method="bonferroni")
            for (g1, g2), p, ap in zip(pairs, raw_p, adj_p):
                posthoc_rows.append({
                    "组1": GROUP_LABELS[g1], "组2": GROUP_LABELS[g2],
                    "p(原始, Mann-Whitney)": round(p, 4),
                    "p(Bonferroni校正)": round(ap, 4), "显著": ap < ALPHA,
                })
    result["事后两两比较"] = posthoc_rows
    return result


def test_categorical(df: pd.DataFrame, col: str) -> dict:
    table = pd.crosstab(df["组别标签"], df[col])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    min_expected = expected.min()

    exact_note = ""
    if min_expected < 5:
        # 期望频数偏低时用置换检验(蒙特卡洛)给出稳健 p 值作为交叉验证
        rng = np.random.default_rng(0)
        observed_chi2 = chi2
        group_arr = df["组别标签"].values
        cat_arr = df[col].values
        n_perm = 2000
        count = 0
        for _ in range(n_perm):
            perm_group = rng.permutation(group_arr)
            perm_table = pd.crosstab(perm_group, cat_arr)
            perm_chi2, *_ = stats.chi2_contingency(perm_table)
            if perm_chi2 >= observed_chi2:
                count += 1
        perm_p = (count + 1) / (n_perm + 1)
        exact_note = f"最小期望频数={min_expected:.2f}<5，蒙特卡洛置换p={perm_p:.4f}"

    n = table.values.sum()
    k = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else np.nan

    return {
        "检验对象": col, "检验方法": "卡方独立性检验",
        "卡方值": round(chi2, 3), "自由度": dof, "p值": round(p, 4),
        "最小期望频数": round(min_expected, 2), "备注": exact_note,
        "Cramer's V(效应量)": round(cramers_v, 3),
        "显著(未校正)": p < ALPHA,
        "列联表": table,
    }


def check_hospital_group_balance(df: pd.DataFrame) -> dict:
    """核查医院与组别分配是否独立（应不显著，否则说明分组在两院不均衡）"""
    table = pd.crosstab(df["医院"], df["组别标签"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"检验对象": "医院×组别(分配均衡性核查)", "检验方法": "卡方独立性检验",
            "卡方值": round(chi2, 3), "自由度": dof, "p值": round(p, 4),
            "显著(未校正)": p < ALPHA, "列联表": table}


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
def plot_age_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = split_age_by_group(df)
    ax.boxplot(data, tick_labels=list(GROUP_LABELS.values()), showmeans=True)
    ax.set_ylabel("年龄（岁）")
    ax.set_title("各组年龄分布")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "age_boxplot.png", dpi=200)
    plt.close(fig)


def plot_age_test_pvalues(df: pd.DataFrame):
    """可视化年龄的正态性(Shapiro)/方差齐性(Levene)/组间差异(ANOVA、Kruskal-Wallis)检验 p 值。

    见 compute_age_diagnostics() 说明：此处需要未四舍五入的原始 p 值。
    """
    groups = [s.values for s in split_age_by_group(df)]
    diag = compute_age_diagnostics(groups)
    shapiro_p, levene_p, anova_p, kw_p = (
        diag["shapiro_p"], diag["levene_p"], diag["anova_p"], diag["kw_p"],
    )

    labels = [f"Shapiro\n({label})" for label in GROUP_LABELS.values()]
    labels += ["Levene", "ANOVA", "Kruskal-\nWallis"]
    values = shapiro_p + [levene_p, anova_p, kw_p]

    y_min = min(v for v in values if v > 0) / 5
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["#788C5D" if v > ALPHA else "#B04A4A" for v in values]
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylim(y_min, 1)
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1, label=f"α = {ALPHA}", zorder=3)
    for bar, v in zip(bars, values):
        label = f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), y_min) * 1.15,
                 label, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("p 值（对数坐标）")
    ax.set_title("年龄：正态性/方差齐性/组间差异检验 p 值")
    ax.legend(loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "age_test_pvalues.png", dpi=200)
    plt.close(fig)


def plot_categorical_stack(df: pd.DataFrame, col: str, filename: str, title: str):
    table = pd.crosstab(df["组别标签"], df[col], normalize="index") * 100
    table = table[sorted(table.columns, key=lambda c: (c == "未知", c))]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(table))
    for cat in table.columns:
        ax.bar(table.index, table[cat], bottom=bottom, label=cat)
        bottom += table[cat].values
    ax.set_ylabel("占比 (%)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(table.columns))
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


def plot_hospital_group_heatmap(df: pd.DataFrame):
    table = pd.crosstab(df["医院"], df["组别标签"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(table.values, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, table.values[i, j], ha="center", va="center")
    ax.set_title("医院 × 组别 样本量核查")
    fig.colorbar(im, ax=ax, shrink=0.8, label="样本量")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hospital_group_heatmap.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    df = build_dataset()
    print(f"合并后总样本量: {len(df)}  (各组: {df['组别标签'].value_counts().to_dict()})")

    # ---- Table 1 ----
    age_summary = summarize_age(df)
    marital_summary = summarize_categorical(df, "婚姻状况")
    med_summary = summarize_categorical(df, "既往用药")
    dep_summary = summarize_categorical(df, "抑郁程度")

    table1_path = OUT_DIR / "table1_baseline_summary.csv"
    with open(table1_path, "w", encoding="utf-8-sig") as f:
        f.write("### 年龄 ###\n")
        age_summary.to_csv(f, index=False)
        f.write("\n### 婚姻状况 ###\n")
        marital_summary.to_csv(f, index=False)
        f.write("\n### 既往用药情况 ###\n")
        med_summary.to_csv(f, index=False)
        f.write("\n### 初始抑郁程度 ###\n")
        dep_summary.to_csv(f, index=False)
    print(f"已保存基线特征汇总表 -> {table1_path}")

    # ---- 假设检验 ----
    age_test = test_age(df)
    marital_test = test_categorical(df, "婚姻状况")
    med_test = test_categorical(df, "既往用药")
    dep_test = test_categorical(df, "抑郁程度")
    hosp_balance = check_hospital_group_balance(df)

    main_tests = [age_test, marital_test, med_test, dep_test]
    raw_p = [t["推荐方法p值"] if "推荐方法p值" in t else t["p值"] for t in main_tests]
    _, adj_p, _, _ = multipletests(raw_p, alpha=ALPHA, method="fdr_bh")
    for t, ap in zip(main_tests, adj_p):
        t["p值(FDR校正)"] = round(ap, 4)
        t["显著(FDR校正)"] = ap < ALPHA

    summary_rows = []
    for t in main_tests:
        p_col = "推荐方法p值" if "推荐方法p值" in t else "p值"
        summary_rows.append({
            "检验对象": t["检验对象"], "方法": t.get("推荐方法", t["检验方法"]),
            "统计量": t.get("ANOVA F") if "推荐方法" in t and t["推荐方法"] == "ANOVA"
                       else t.get("Kruskal-Wallis H", t.get("卡方值")),
            "原始p值": t[p_col], "FDR校正p值": t["p值(FDR校正)"],
            "0.05水平下是否显著": t["显著(FDR校正)"],
        })
    summary_df = pd.DataFrame(summary_rows)

    test_results_path = OUT_DIR / "test_results.csv"
    with open(test_results_path, "w", encoding="utf-8-sig") as f:
        f.write("### 假设检验总览（含多重比较FDR校正） ###\n")
        summary_df.to_csv(f, index=False)

        f.write("\n### 年龄：正态性/方差齐性诊断 ###\n")
        pd.DataFrame([{
            "各组Shapiro p": age_test["正态性(各组Shapiro p)"],
            "Levene p": age_test["方差齐性(Levene p)"],
            "ANOVA F": age_test["ANOVA F"], "ANOVA p": age_test["ANOVA p"],
            "Kruskal H": age_test["Kruskal-Wallis H"], "Kruskal p": age_test["Kruskal-Wallis p"],
        }]).to_csv(f, index=False)
        if age_test["事后两两比较"]:
            f.write("\n### 年龄：事后两两比较 ###\n")
            pd.DataFrame(age_test["事后两两比较"]).to_csv(f, index=False)

        for t in [marital_test, med_test, dep_test]:
            f.write(f"\n### {t['检验对象']}：列联表 ###\n")
            t["列联表"].to_csv(f)
            f.write(f"\n### {t['检验对象']}：卡方检验详情 ###\n")
            detail = {k: v for k, v in t.items() if k not in ("列联表",)}
            pd.DataFrame([detail]).to_csv(f, index=False)

        f.write("\n### 医院×组别 分配均衡性核查 ###\n")
        hosp_balance["列联表"].to_csv(f)
        pd.DataFrame([{k: v for k, v in hosp_balance.items() if k != "列联表"}]).to_csv(f, index=False)

    print(f"已保存假设检验结果 -> {test_results_path}")

    # ---- 图表 ----
    plot_age_box(df)
    plot_age_test_pvalues(df)
    plot_categorical_stack(df, "婚姻状况", "marital_status_bar.png", "各组婚姻状况分布")
    plot_categorical_stack(df, "既往用药", "prior_use_bar.png", "各组既往抗抑郁药使用情况分布")
    plot_categorical_stack(df, "抑郁程度", "depression_severity_bar.png", "各组初始抑郁程度分布")
    plot_hospital_group_heatmap(df)
    print(f"已保存图表 -> {OUT_DIR}")

    # ---- 控制台摘要 ----
    print("\n=== 假设检验总览 ===")
    print(summary_df.to_string(index=False))
    print(f"\n医院×组别均衡性核查 p = {hosp_balance['p值']}"
          f"（{'不均衡，需留意' if hosp_balance['显著(未校正)'] else '均衡，符合预期'}）")


if __name__ == "__main__":
    main()
