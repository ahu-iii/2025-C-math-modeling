"""
问题2：基线因素（婚姻状况、既往用药史、初始抑郁程度）对疗效与主诉的影响。

按药物分层建模（组1=药物C对照 / 组2=药物A / 组3=药物B），不构建"药物×因素"
交互项（组别与药物完全共线，见 docs/adr/0006）。每组单独跑一套回归。

结局变量（从附件2 构造，缺席附件2 的受试者=全程无不适，见 docs/adr/0001、0003、0005）：
  · 非缓解  D12=1           —— 罕见结局(3-5%)，分层后事件少 → Firth 惩罚 Logistic（外加合并模型作锚）
  · 任何主诉 任一时点 D_t=1  —— 高发结局(~32%) → 普通 Logistic
  · 总症状负担 ΣB_t          —— 计数结局 → 负二项回归（过离散/零膨胀）

自变量：婚姻状况(参照=已婚) + 既往用药(参照=无用药史) + 抑郁程度(有序 1/2/3 + 未知指示) + 年龄(每10岁)

数据口径提醒（暂用默认，待团队确认，详见 docs/建模待决问题.md、docs/adr/0010）：
  · 附件2 二院/组2 存在整块重复录入（194 人），按 (医院,序号) 合并、症状标记取并集(union)。
  · 缓解=D12=0（单点，open-Q1 默认）；复发仅描述、不进入 Q2 回归（分层后事件过少）。

输出：
  output/problem2_or_table.csv     - 各结局×各组 的 OR/IRR + 95%CI + p（长表）
  output/problem2_diagnostics.csv  - 队列核对、各组事件数/EPV、分离(separation)标记、缓解/复发描述率
  output/problem2_remission_forest.png - 非缓解 OR 森林图（C/A/B/合并）
  output/problem2_complaint_burden.png - 任何主诉 OR 与 症状负担 IRR 森林图
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
F1 = ROOT / "附件1：两个院临床受试者及抑郁症的基本数据.xlsx"
F2 = ROOT / "附件2：两个医院随访的抗抑郁药使用后主诉情况.xlsx"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

GROUPS = {1: "药物C(对照)", 2: "药物A", 3: "药物B"}
TIMES = [1, 3, 6, 12]
SYMPTOMS = ["失访", "有自杀倾向", "副作用导致停药", "失眠", "脱发", "激素水平异常", "嗜睡", "便秘"]
# 7 类实质性不适；失访是"缺失/失访"而非症状，D_t 不计入(ADR-0001)、B_t/症状负担也不计入(ADR-0005)。
# 已核实：附件2 现成的"是否出现不适症状"栏 == 这 7 类之和（4 时点 0 误差），佐证 失访 应排除。
SUBST = SYMPTOMS[1:]
ALPHA = 0.05

# 附件1 列（与 problem1_baseline.py 一致）
COL1 = ["序号", "组别", "年龄",
        "未婚", "已婚", "离异", "丧偶",
        "无用药史", "使用过抗抑郁药", "其它用药史",
        "轻度", "中度", "重度"]


# --------------------------------------------------------------------------
# 中文字体
# --------------------------------------------------------------------------
def setup_cjk_font():
    for path in ["/System/Library/Fonts/Supplemental/Heiti SC.ttc",
                 "/System/Library/Fonts/Supplemental/STHeiti Light.ttc",
                 "/System/Library/Fonts/Supplemental/Songti.ttc"]:
        if Path(path).exists():
            matplotlib.font_manager.fontManager.addfont(path)
            plt.rcParams["font.sans-serif"] = [
                matplotlib.font_manager.FontProperties(fname=path).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return
    warnings.warn("未找到中文字体，图中文字可能显示为方框。")


setup_cjk_font()


# --------------------------------------------------------------------------
# 数据加载与结局构造
# --------------------------------------------------------------------------
def onehot_to_category(df, cols, out_name):
    """一组 one-hot 列坍缩为单个分类变量；全空记为'未知'（与 problem1 口径一致）。

    已知边界情形：附件1 有 3/3149 受试者同一组内命中>1 个 one-hot（录入笔误），此时
    idxmax 取 cols 中靠前者（如 已婚+丧偶→已婚）；样本量极小、不影响结论，保持与 problem1 一致。"""
    sub = df[cols]
    cat = sub.fillna(0).idxmax(axis=1).where(sub.eq(1).any(axis=1), other="未知")
    cat.name = out_name
    return cat


def load_baseline():
    frames = []
    for sheet, hosp in [("一院临床受试者及抑郁症的基本数据", "一院"),
                        ("二院临床受试者及抑郁症的基本数据", "二院")]:
        d = pd.read_excel(F1, sheet_name=sheet, header=None, skiprows=2,
                          usecols=range(len(COL1)), names=COL1).dropna(how="all")
        d[COL1[3:]] = d[COL1[3:]].apply(pd.to_numeric, errors="coerce")
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
    df["曾缓解"] = ((df["D3"] == 0) | (df["D6"] == 0)).astype(int)
    df["复发"] = (((df["D3"] == 0) | (df["D6"] == 0)) & (df["D12"] == 1)).astype(int)

    df["年龄"] = df["年龄"].fillna(df["年龄"].median())
    return df, {"附件2原始行数": n_raw, "去重删除行数": n_dup, "建模队列N": len(df)}


# --------------------------------------------------------------------------
# 设计矩阵
# --------------------------------------------------------------------------
DEP_MAP = {"轻度": 1, "中度": 2, "重度": 3}


def build_design(df, pooled=False):
    """构造回归自变量矩阵（不含截距）。婚姻/用药用哑变量，抑郁用有序编码+未知指示。"""
    X = pd.DataFrame(index=df.index)
    dep = df["抑郁程度"].map(DEP_MAP)
    X["抑郁_未知"] = dep.isna().astype(float)                    # 未知的哑标记
    X["抑郁_程度"] = dep.fillna(0).astype(float)                 # 有序 1/2/3，未知置0由上面标记吸收
    X["年龄_每10岁"] = df["年龄"].astype(float) / 10.0

    mar = pd.get_dummies(df["婚姻状况"], prefix="婚姻").drop(columns=["婚姻_已婚"], errors="ignore")
    med = pd.get_dummies(df["既往用药"], prefix="用药").drop(columns=["用药_无用药史"], errors="ignore")
    X = pd.concat([X, mar.astype(float), med.astype(float)], axis=1)
    if pooled:
        drug = pd.get_dummies(df["组别"].map({1: "C", 2: "A", 3: "B"}), prefix="药物")
        X = pd.concat([X, drug.drop(columns=["药物_C"], errors="ignore").astype(float)], axis=1)

    X = X.loc[:, X.nunique() > 1]                                # 丢弃零方差列（分层后某类别可能不存在）
    return X


# --------------------------------------------------------------------------
# Firth 惩罚 Logistic（Jeffreys 先验，处理罕见结局/完全分离）
# --------------------------------------------------------------------------
def _penalized_loglik(X, y, beta):
    eta = X @ beta
    ll = np.sum(y * eta - np.logaddexp(0, eta))
    W = expit(eta) * (1 - expit(eta))
    sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    return ll + 0.5 * logdet if sign > 0 else -np.inf


def firth_logit(X, y, max_iter=200, tol=1e-8):
    """Firth 惩罚 Logistic 回归（Newton-Raphson + 惩罚似然步长折半）。返回 (beta, se)。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    ll = _penalized_loglik(X, y, beta)
    for _ in range(max_iter):
        eta = X @ beta
        pi = expit(eta)
        W = pi * (1 - pi)
        info_inv = np.linalg.pinv(X.T @ (X * W[:, None]))       # Fisher 信息逆
        h = W * np.einsum("ij,ij->i", X @ info_inv, X)          # 帽子矩阵对角
        score = X.T @ (y - pi + h * (0.5 - pi))                 # Firth 修正得分
        step = info_inv @ score
        f = 1.0                                                 # 步长折半保证惩罚似然单调上升
        while f > 1e-6:
            cand = beta + f * step
            ll_new = _penalized_loglik(X, y, cand)
            if ll_new >= ll - 1e-12:
                break
            f *= 0.5
        beta, prev_ll = beta + f * step, ll
        ll = ll_new
        if np.max(np.abs(f * step)) < tol:
            break
    pi = expit(X @ beta)
    cov = np.linalg.pinv(X.T @ (X * (pi * (1 - pi))[:, None]))
    return beta, np.sqrt(np.diag(cov))


def firth_profile_ci(X, y, beta, se, level=0.95):
    """Firth 系数的惩罚 profile-likelihood 置信区间 + LR 检验 p 值。

    这是 Firth 点估计的标准配套推断（logistf/Heinze-Schemper）：Wald 区间在近完全分离
    的格子里（本题 B 组 抑郁_未知）校准很差，profile-likelihood 区间在此更可靠，且当上界
    实际发散时如实返回 inf（而非 Wald 那样给出虚假的有限上界）。返回 (lo, hi, pval)，均为
    原始系数尺度（取 exp 后即 OR 界）。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    pll_max = _penalized_loglik(X, y, beta)
    crit = stats.chi2.ppf(level, 1)
    lo, hi, pval = np.full(p, -np.inf), np.full(p, np.inf), np.full(p, np.nan)

    def constrained_pll(j, val, x0):
        free = [k for k in range(p) if k != j]

        def neg(bf):
            b = np.empty(p)
            b[j], b[free] = val, bf
            return -_penalized_loglik(X, y, b)

        return -minimize(neg, x0, method="BFGS").fun

    for j in range(p):
        x0 = beta[[k for k in range(p) if k != j]]
        dev = lambda v: 2 * (pll_max - constrained_pll(j, v, x0))   # 惩罚偏差差
        pval[j] = float(1 - stats.chi2.cdf(max(dev(0.0), 0.0), 1))  # LR 检验 (beta_j=0)
        step = max(se[j], 0.5)
        for sign, store in ((+1, "hi"), (-1, "lo")):                # 向两侧扩展找 dev=crit
            edge = beta[j]
            for _ in range(80):
                edge += sign * step
                if dev(edge) > crit:
                    root = brentq(lambda v: dev(v) - crit, beta[j], edge)
                    (hi if store == "hi" else lo).__setitem__(j, root)
                    break
    return lo, hi, pval


# --------------------------------------------------------------------------
# 统一的结果整理（OR/IRR + 95%CI + p）
# --------------------------------------------------------------------------
def _tidy(names, coef, outcome, group, method, n, n_event, effect,
          ci_lo, ci_hi, pvals):
    """整理成长表。ci_lo/ci_hi/pvals 为系数尺度（Wald 或 profile-likelihood 均可）。"""
    rows = []
    for name, b, lo, hi, pv in zip(names, coef, ci_lo, ci_hi, pvals):
        if name == "const":
            continue
        rows.append({
            "结局": outcome, "组别": group, "方法": method, "变量": name,
            effect: round(float(np.exp(b)), 3),
            "CI下": round(float(np.exp(lo)), 3),
            "CI上": float("inf") if np.isinf(hi) else round(float(np.exp(hi)), 3),
            "p": round(float(pv), 4),
            "n": n, "事件数/计数": n_event, "效应类型": effect,
        })
    return rows


def _wald(coef, se):
    z = np.asarray(coef) / np.asarray(se)
    return coef - 1.96 * se, coef + 1.96 * se, 2 * (1 - stats.norm.cdf(np.abs(z)))


def fit_firth(df, outcome, group_label):
    X = build_design(df)
    Xc = sm.add_constant(X, has_constant="add")
    beta, se = firth_logit(Xc.values, df[outcome].values)
    lo, hi, pv = firth_profile_ci(Xc.values, df[outcome].values, beta, se)  # profile-likelihood
    return _tidy(list(Xc.columns), beta, outcome, group_label,
                 "Firth Logistic(profile-CI)", len(df), int(df[outcome].sum()), "OR", lo, hi, pv)


def fit_logit(df, outcome, group_label):
    X = build_design(df)
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.Logit(df[outcome].values, Xc.values).fit(disp=0, maxiter=200)
    lo, hi, pv = _wald(res.params, res.bse)
    return _tidy(list(Xc.columns), res.params, outcome, group_label,
                 "Logistic", len(df), int(df[outcome].sum()), "OR", lo, hi, pv)


def fit_nb(df, outcome, group_label):
    X = build_design(df)
    Xc = sm.add_constant(X, has_constant="add")
    try:
        res = sm.NegativeBinomial(df[outcome].values, Xc.values).fit(disp=0, maxiter=200)
        method = "负二项回归"
    except Exception:
        res = sm.GLM(df[outcome].values, Xc.values, family=sm.families.Poisson()).fit()
        method = "Poisson回归(NB未收敛回退)"
    k = len(Xc.columns)                                         # NB 末位是 alpha，非回归系数
    coef, se = np.asarray(res.params)[:k], np.asarray(res.bse)[:k]
    lo, hi, pv = _wald(coef, se)
    return _tidy(list(Xc.columns), coef, outcome, group_label, method,
                 len(df), int(df[outcome].sum()), "IRR", lo, hi, pv)


def fit_pooled_logit(df, outcome):
    X = build_design(df, pooled=True)
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.Logit(df[outcome].values, Xc.values).fit(disp=0, maxiter=200)
    lo, hi, pv = _wald(res.params, res.bse)
    return _tidy(list(Xc.columns), res.params, outcome, "合并(锚)",
                 "Logistic(合并+药物主效应)", len(df), int(df[outcome].sum()), "OR", lo, hi, pv)


# --------------------------------------------------------------------------
# 诊断：EPV、分离标记、缓解/复发描述率
# --------------------------------------------------------------------------
def diagnostics(df):
    rows = []
    for g, label in GROUPS.items():
        sub = df[df["组别"] == g]
        X = build_design(sub)
        n_param = X.shape[1] + 1                                # + 截距
        n_event = int(sub["非缓解"].sum())
        # 分离标记：任一哑变量类别内非缓解事件数为 0
        sep = [c for c in X.columns if set(X[c].unique()) <= {0.0, 1.0}
               and sub.loc[X[c] == 1, "非缓解"].sum() == 0 and (X[c] == 1).sum() > 0]
        rows.append({
            "组别": label, "n": len(sub),
            "非缓解事件数": n_event, "非缓解率%": round(100 * sub["非缓解"].mean(), 1),
            "回归参数数": n_param, "EPV(事件/参数)": round(n_event / n_param, 1),
            "分离风险类别(非缓解=0)": "; ".join(sep) if sep else "无",
            "缓解率%(D12=0)": round(100 * (1 - sub["非缓解"].mean()), 1),
            "曾缓解n": int(sub["曾缓解"].sum()), "复发n": int(sub["复发"].sum()),
            "复发率%(/曾缓解)": round(100 * sub["复发"].sum() / sub["曾缓解"].sum(), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 可视化：森林图
# --------------------------------------------------------------------------
GROUP_COLORS = {"药物C(对照)": "#5C7CA3", "药物A": "#D97757", "药物B": "#788C5D", "合并(锚)": "#141413"}
GROUP_SHORT = {"药物C(对照)": "C", "药物A": "A", "药物B": "B", "合并(锚)": "合并"}


def _forest(ax, res_df, effect, title):
    """在 ax 上画森林图：y=变量×组别，x=效应量(对数)，须线=95%CI。"""
    order = list(dict.fromkeys(res_df["变量"]))                 # 保持变量出现顺序
    groups = list(dict.fromkeys(res_df["组别"]))
    y = 0
    yticks, ylabels = [], []
    for var in order:
        for grp in groups:
            r = res_df[(res_df["变量"] == var) & (res_df["组别"] == grp)]
            if r.empty:
                continue
            r = r.iloc[0]
            color = GROUP_COLORS.get(grp, "#87867F")
            ax.plot([r["CI下"], r["CI上"]], [y, y], color=color, lw=1.5, zorder=2)
            ax.plot(r[effect], y, "o", color=color, ms=5, zorder=3)
            yticks.append(y)
            ylabels.append(f"{var} · {GROUP_SHORT.get(grp, grp)}")
            y -= 1
        y -= 0.5
    ax.axvline(1.0, color="#87867F", ls="--", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel(f"{effect}（对数坐标，虚线=1 无效应）")
    ax.set_title(title)


def plot_remission_forest(res_df):
    sub = res_df[res_df["结局"] == "非缓解"].copy()
    fig, ax = plt.subplots(figsize=(8, 10))
    _forest(ax, sub, "OR", "非缓解的影响因素（各药物分层 + 合并锚定）\nOR>1 = 该特征更难缓解")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "problem2_remission_forest.png", dpi=200)
    plt.close(fig)


def plot_complaint_burden(res_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    _forest(axes[0], res_df[res_df["结局"] == "任何主诉"], "OR",
            "出现任何主诉的影响因素（分层）\nOR>1 = 该特征更易出现主诉")
    _forest(axes[1], res_df[res_df["结局"] == "总症状负担"], "IRR",
            "总症状负担的影响因素（分层）\nIRR>1 = 该特征症状负担更高")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "problem2_complaint_burden.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    df, meta = build_cohort()
    print(f"附件2 原始 {meta['附件2原始行数']} 行 → 去重删除 {meta['去重删除行数']} 行；"
          f"建模队列 N={meta['建模队列N']}（应=3149）")
    print(f"各组: {df['组别'].map(GROUPS).value_counts().to_dict()}")

    diag = diagnostics(df)
    print("\n=== 分层诊断（EPV / 分离风险 / 缓解复发描述率）===")
    print(diag.to_string(index=False))

    results = []
    for g, label in GROUPS.items():
        sub = df[df["组别"] == g]
        results += fit_firth(sub, "非缓解", label)              # 罕见结局 → Firth
        results += fit_logit(sub, "任何主诉", label)            # 高发结局 → 普通 Logistic
        results += fit_nb(sub, "总症状负担", label)             # 计数结局 → 负二项
    results += fit_pooled_logit(df, "非缓解")                   # 合并锚（稳定 B 组的脆弱估计）
    res_df = pd.DataFrame(results)

    # Benjamini-Hochberg (FDR) 多重比较校正——与问题1/建模方案§七 口径一致。
    # 校正族 = 每个回归模型内部的协变量检验（同一 结局×组别×方法 的所有系数）。
    res_df["p_FDR"] = np.nan
    for _, idx in res_df.groupby(["结局", "组别", "方法"]).groups.items():
        res_df.loc[idx, "p_FDR"] = multipletests(res_df.loc[idx, "p"], method="fdr_bh")[1]
    res_df["p_FDR"] = res_df["p_FDR"].round(4)
    res_df["显著_FDR"] = res_df["p_FDR"] < ALPHA

    out = OUT_DIR / "problem2_or_table.csv"
    res_df.to_csv(out, index=False, encoding="utf-8-sig")
    diag.to_csv(OUT_DIR / "problem2_diagnostics.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存效应量表 -> {out}")
    print(f"已保存诊断表 -> {OUT_DIR / 'problem2_diagnostics.csv'}")

    plot_remission_forest(res_df)
    plot_complaint_burden(res_df)
    print(f"已保存森林图 -> {OUT_DIR}")

    sig = res_df[res_df["显著_FDR"]]
    print(f"\n=== FDR 校正后显著因素（p_FDR<{ALPHA}，共 {len(sig)} 项）===")
    if not sig.empty:
        show = sig.assign(效应量=[r[r["效应类型"]] for _, r in sig.iterrows()])
        print(show[["结局", "组别", "变量", "效应类型", "效应量", "CI下", "CI上", "p", "p_FDR"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
