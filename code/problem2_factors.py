"""
问题2：基线因素（婚姻状况、既往用药史、初始抑郁程度）对疗效与主诉的影响。

按药物分层建模（组1=药物C对照 / 组2=药物A / 组3=药物B），题目要求不考虑因素间交互作用，
故主分析不构建“药物×因素”交互项，每组单独跑一套回归。交互项在合并模型中数学上可识别，
只是未作为本题主分析；分层系数差异因此只作描述，不当作正式效应异质性检验（见 ADR-0006）。

结局变量（从附件2 构造，缺席附件2 的受试者=全程无不适，见 docs/adr/0001、0003、0005）：
  · 非缓解  D12=1           —— “第12月仍有不适”的主诉代理结局，不等同临床抑郁未缓解；
                               事件罕见(3.5%-5.3%) → Firth 惩罚 Logistic（外加合并模型作锚）
  · 任何主诉 任一时点 D_t=1  —— 高发结局(~33%) → 普通 Logistic
  · 总症状负担 ΣB_t          —— 计数结局 → 负二项回归（过离散/零膨胀）

自变量：婚姻状况(参照=已婚) + 既往用药(参照=无用药史) + 抑郁程度(有序 1/2/3 + 未知指示) + 年龄(每10岁)

数据口径提醒（均已定案，详见 docs/adr/0010、0003）：
  · 附件2 二院/组2 存在整块重复录入（194 人），按 (医院,序号) 合并、症状标记取并集(union)。
  · 缓解=D12=0（单点，ADR-0003 已定案）；复发仅描述、不进入 Q2 回归（分层后事件过少）。

输出（论文插图与诊断输出的分级见 docs/adr/0011）：
  output/problem2_or_table.csv     - 各结局×各组 的 OR/IRR + 95%CI + p（含全局与模型内 FDR）
  output/problem2_diagnostics.csv  - 队列核对、各组事件数/EPV、分离(separation)标记、缓解/复发描述率
  output/problem2_severity_table.csv   - 抑郁程度×药物 的观测率 + Wilson CI（图5 底层数值）
  output/problem2_severity_gradient.png - 抑郁程度→结局 的观测梯度（图5）
  output/problem2_remission_forest.png  - 非缓解 OR 森林图（图6；C/A/B/合并）
  output/problem2_complaint_burden.png  - 任何主诉 OR 与 症状负担 IRR 森林图（图7）
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.special import expit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from common import ALPHA, GROUPS, OUT_DIR, build_cohort

warnings.filterwarnings("ignore")


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
            "第12月无主诉率%(代理缓解)": round(100 * (1 - sub["非缓解"].mean()), 1),
            "曾缓解n": int(sub["曾缓解"].sum()), "复发n": int(sub["复发"].sum()),
            "复发率%(/曾缓解)": round(100 * sub["复发"].sum() / sub["曾缓解"].sum(), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 可视化：森林图
# --------------------------------------------------------------------------
GROUP_COLORS = {"药物C(对照)": "#5C7CA3", "药物A": "#D97757", "药物B": "#788C5D", "合并(锚)": "#141413"}
GROUP_SHORT = {"药物C(对照)": "C", "药物A": "A", "药物B": "B", "合并(锚)": "合并"}

# 森林图的变量排序与人读标签。前段为可解读的实质因素，NUISANCE 为缺失指示项——
# 它们是有序编码/哑变量的技术性 nuisance 项（把未知者程度置 0 后另设标记），系数不代表
# "未知比轻度差 N 倍"，故在图中单独成块、灰显，避免被当作结论读（见 docs/问题2_实现说明.md §七.2）。
VAR_LABEL = {
    "抑郁_程度": "初始抑郁程度（每加重一级）",
    "婚姻_未婚": "婚姻：未婚（vs 已婚）",
    "婚姻_离异": "婚姻：离异（vs 已婚）",
    "婚姻_丧偶": "婚姻：丧偶（vs 已婚）",
    "用药_使用过抗抑郁药": "既往：使用过抗抑郁药（vs 无用药史）",
    "用药_其它用药史": "既往：其它用药史（vs 无用药史）",
    "年龄_每10岁": "年龄（每增 10 岁）",
    "药物_A": "药物 A（vs 对照药 C）",
    "药物_B": "药物 B（vs 对照药 C）",
    "抑郁_未知": "［抑郁程度缺失指示］",
    "婚姻_未知": "［婚姻状况缺失指示］",
}
VAR_ORDER = ["抑郁_程度", "婚姻_未婚", "婚姻_离异", "婚姻_丧偶",
             "用药_使用过抗抑郁药", "用药_其它用药史", "年龄_每10岁",
             "药物_A", "药物_B", "抑郁_未知", "婚姻_未知"]
NUISANCE = {"抑郁_未知", "婚姻_未知"}


def _forest(ax, res_df, effect, title):
    """在 ax 上画森林图：y=变量×组别，x=效应量(对数)，须线=95%CI。

    实质因素与"缺失指示"nuisance 项分块绘制（后者灰显）；坐标轴范围由实质因素的
    CI 决定，越界的 CI 端点以箭头示意（否则 B 组分离格子的 CI 上界会把横轴拉到 10^2
    量级，实质因素全被压成一团）。FDR 校正后显著者用实心点，不显著者用空心点。"""
    present = [v for v in VAR_ORDER if v in set(res_df["变量"])]
    groups = [g for g in GROUP_COLORS if g in set(res_df["组别"])]
    subst = [v for v in present if v not in NUISANCE]

    # 轴范围只由实质因素决定，nuisance 的发散 CI 不参与
    fin = res_df[res_df["变量"].isin(subst)]
    vals = np.concatenate([fin[effect].values, fin["CI下"].values, fin["CI上"].values])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lo_lim, hi_lim = vals.min() / 1.6, vals.max() * 1.6

    y = 0.0
    yticks, ylabels, nuis_ys = [], [], []
    for var in present:
        is_nuis = var in NUISANCE
        for grp in groups:
            r = res_df[(res_df["变量"] == var) & (res_df["组别"] == grp)]
            if r.empty:
                continue
            r = r.iloc[0]
            color = "#87867F" if is_nuis else GROUP_COLORS[grp]
            lo, hi = max(r["CI下"], lo_lim), min(r["CI上"], hi_lim)
            ax.plot([lo, hi], [y, y], color=color, lw=1.4, zorder=2,
                    alpha=0.55 if is_nuis else 1.0)
            for bound, clipped, direc in ((r["CI下"], lo, "left"), (r["CI上"], hi, "right")):
                if not (lo_lim <= bound <= hi_lim):              # CI 越界 → 画箭头示意发散
                    ax.plot(clipped, y, marker=f"{'<' if direc == 'left' else '>'}",
                            color=color, ms=4, zorder=3, alpha=0.55 if is_nuis else 1.0)
            sig = bool(r.get("显著_FDR", False))
            ax.plot(r[effect], y, "o", ms=5.5, zorder=4, color=color,
                    mfc=color if sig else "white", mew=1.4,
                    alpha=0.55 if is_nuis else 1.0)
            yticks.append(y)
            ylabels.append(f"{VAR_LABEL.get(var, var)} · {GROUP_SHORT.get(grp, grp)}")
            if is_nuis:
                nuis_ys.append(y)
            y -= 1
        y -= 0.6

    if nuis_ys:                                                  # 给 nuisance 块加底纹并标注
        ax.axhspan(min(nuis_ys) - 0.6, max(nuis_ys) + 0.6, color="#F0EEE6", zorder=0)
        ax.text(lo_lim * 1.05, max(nuis_ys) + 0.5, "缺失指示项（技术性 nuisance，不作解读）",
                fontsize=6.5, color="#87867F", va="bottom", ha="left")

    ax.axvline(1.0, color="#87867F", ls="--", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(lo_lim, hi_lim)
    # 只保留落在范围内的"整齐" OR 刻度，并关掉次刻度——否则 log 轴的默认次刻度标签会互相重叠
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo_lim <= t <= hi_lim]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_ylim(y + 0.4, 0.8)
    ax.set_xlabel(f"{effect}（对数坐标；虚线 = 1，即无效应）", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _forest_legend(fig, groups, note):
    handles = [plt.Line2D([], [], color=GROUP_COLORS[g], marker="o", ls="-", ms=5,
                          label=GROUP_SHORT[g]) for g in groups]
    handles += [
        plt.Line2D([], [], color="#141413", marker="o", ls="none", ms=5.5, mfc="#141413",
                   label="FDR 校正后显著"),
        plt.Line2D([], [], color="#141413", marker="o", ls="none", ms=5.5, mfc="white",
                   mew=1.4, label="不显著"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.048, note, ha="center", fontsize=7, color="#87867F")


def plot_remission_forest(res_df):
    sub = res_df[res_df["结局"] == "非缓解"].copy()
    fig, ax = plt.subplots(figsize=(8.5, 9))
    _forest(ax, sub, "OR", "非缓解（第 12 月仍有不适）的影响因素\n各药物分层估计 + 合并锚定；OR>1 = 该特征更难缓解")
    _forest_legend(fig, ["药物C(对照)", "药物A", "药物B", "合并(锚)"],
                   "分层估计用 Firth 惩罚 Logistic（profile-likelihood CI），合并锚用含药物主效应的 Logistic；箭头表示 CI 超出坐标范围")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(OUT_DIR / "problem2_remission_forest.png", dpi=200)
    plt.close(fig)


def plot_complaint_burden(res_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    _forest(axes[0], res_df[res_df["结局"] == "任何主诉"], "OR",
            "出现任何主诉（任一随访时点有不适）的影响因素\nOR>1 = 该特征更易出现主诉")
    _forest(axes[1], res_df[res_df["结局"] == "总症状负担"], "IRR",
            "总症状负担（四时点症状数之和）的影响因素\nIRR>1 = 该特征症状负担更高")
    _forest_legend(fig, ["药物C(对照)", "药物A", "药物B"],
                   "左：Logistic 回归；右：负二项回归。二者均按药物分层，无合并锚")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(OUT_DIR / "problem2_complaint_burden.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 可视化：抑郁程度—结局的剂量反应梯度（描述层，不依赖模型设定）
# --------------------------------------------------------------------------
DEP_LEVELS = ["轻度", "中度", "重度"]
GROUP_LEGEND = {"药物C(对照)": "C（对照药）", "药物A": "A（新药）", "药物B": "B（新药）"}


def plot_severity_gradient(df):
    """三种药下"基线抑郁程度 → 结局"的观测梯度（含 Wilson 95%CI）。

    这是问题2 核心结论的模型无关佐证：回归给的是 OR 这个数值，此图给的是形状——
    三条折线是否单调、谁更陡、有没有交叉。"未知"类别不入图（n=10–20 且为缺失指示，
    非实质严重度等级）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [("非缓解", "非缓解率（第 12 月仍有不适）/ %", "(a) 非缓解率随基线抑郁程度的变化"),
              ("任何主诉", "任何主诉发生率 / %", "(b) 任何主诉发生率随基线抑郁程度的变化")]
    for ax, (outcome, ylab, title) in zip(axes, panels):
        for k, (g, label) in enumerate(GROUPS.items()):
            xs, ys, los, his = [], [], [], []
            for i, d in enumerate(DEP_LEVELS):
                s = df[(df["组别"] == g) & (df["抑郁程度"] == d)][outcome]
                lo, hi = proportion_confint(int(s.sum()), len(s), method="wilson")
                xs.append(i + (k - 1) * 0.06)                    # 轻微错开，避免误差棒重叠
                ys.append(100 * s.mean())
                los.append(100 * (s.mean() - lo))
                his.append(100 * (hi - s.mean()))
            ax.errorbar(xs, ys, yerr=[los, his], color=GROUP_COLORS[label], marker="o",
                        ms=6, lw=1.8, capsize=3, elinewidth=1.2,
                        label=GROUP_LEGEND[label])
        ax.set_xticks(range(len(DEP_LEVELS)))
        ax.set_xticklabels(DEP_LEVELS)
        ax.set_xlabel("基线（初始）抑郁程度", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.4, len(DEP_LEVELS) - 0.6)
        ax.grid(axis="y", color="#D1CFC5", lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.text(0.5, 0.015, "误差棒为 Wilson 95% 置信区间；点为组内观测率，未经模型调整",
             ha="center", fontsize=7, color="#87867F")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT_DIR / "problem2_severity_gradient.png", dpi=200)
    plt.close(fig)


def severity_gradient_table(df):
    """图5 的底层数值（供正文引用与核对）。"""
    rows = []
    for g, label in GROUPS.items():
        for d in DEP_LEVELS:
            for outcome in ["非缓解", "任何主诉"]:
                s = df[(df["组别"] == g) & (df["抑郁程度"] == d)][outcome]
                lo, hi = proportion_confint(int(s.sum()), len(s), method="wilson")
                rows.append({"组别": label, "抑郁程度": d, "结局": outcome,
                             "n": len(s), "事件数": int(s.sum()),
                             "率%": round(100 * s.mean(), 1),
                             "Wilson CI下%": round(100 * lo, 1),
                             "Wilson CI上%": round(100 * hi, 1)})
    return pd.DataFrame(rows)


def severity_encoding_sensitivity(df):
    """合并模型中把抑郁程度改为分类变量，核查有序线性编码是否主导结论。"""
    X = build_design(df, pooled=True).drop(columns=["抑郁_程度", "抑郁_未知"], errors="ignore")
    known = df["抑郁程度"].where(df["抑郁程度"].isin(DEP_MAP))
    dep = pd.get_dummies(known, prefix="抑郁分类").astype(float)
    dep = dep.drop(columns=["抑郁分类_轻度"], errors="ignore")
    X["抑郁_未知"] = known.isna().astype(float)
    X = pd.concat([X, dep], axis=1)
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.Logit(df["非缓解"].values, Xc).fit(disp=False, maxiter=300)

    severity_terms = [c for c in Xc.columns if c.startswith("抑郁分类_")]
    R = np.zeros((len(severity_terms), len(Xc.columns)))
    for i, term in enumerate(severity_terms):
        R[i, list(Xc.columns).index(term)] = 1
    joint = res.wald_test(R, scalar=True)
    ci = res.conf_int()
    rows = [{
        "检验": "抑郁程度分类变量联合Wald",
        "项": "整体",
        "OR": "",
        "CI下": "",
        "CI上": "",
        "统计量": round(float(joint.statistic), 4),
        "df": len(severity_terms),
        "p": round(float(joint.pvalue), 6),
    }]
    for term in severity_terms:
        rows.append({
            "检验": "分类编码系数",
            "项": term.replace("抑郁分类_", "") + " vs 轻度",
            "OR": round(float(np.exp(res.params[term])), 3),
            "CI下": round(float(np.exp(ci.loc[term, 0])), 3),
            "CI上": round(float(np.exp(ci.loc[term, 1])), 3),
            "统计量": "",
            "df": 1,
            "p": round(float(res.pvalues[term]), 6),
        })
    return pd.DataFrame(rows)


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

    # 主口径：对表中全部 86 个系数统一做 BH-FDR，和论文“全部检验”措辞一致。
    # 同时保留“每个模型内部校正”作为敏感性分析，防止把校正族选择藏起来。
    res_df["p_FDR_模型内"] = np.nan
    for _, idx in res_df.groupby(["结局", "组别", "方法"]).groups.items():
        res_df.loc[idx, "p_FDR_模型内"] = multipletests(res_df.loc[idx, "p"], method="fdr_bh")[1]
    res_df["p_FDR_模型内"] = res_df["p_FDR_模型内"].round(4)
    res_df["显著_模型内FDR"] = res_df["p_FDR_模型内"] < ALPHA
    res_df["p_FDR"] = multipletests(res_df["p"], method="fdr_bh")[1].round(4)
    res_df["显著_FDR"] = res_df["p_FDR"] < ALPHA

    sev = severity_gradient_table(df)
    sev_sensitivity = severity_encoding_sensitivity(df)
    print("\n=== 抑郁程度 × 药物 的观测率（图5 底层数值）===")
    print(sev.to_string(index=False))

    out = OUT_DIR / "problem2_or_table.csv"
    res_df.to_csv(out, index=False, encoding="utf-8-sig")
    diag.to_csv(OUT_DIR / "problem2_diagnostics.csv", index=False, encoding="utf-8-sig")
    sev.to_csv(OUT_DIR / "problem2_severity_table.csv", index=False, encoding="utf-8-sig")
    sev_sensitivity.to_csv(OUT_DIR / "problem2_severity_encoding_sensitivity.csv",
                           index=False, encoding="utf-8-sig")
    print(f"\n已保存效应量表 -> {out}")
    print(f"已保存诊断表 -> {OUT_DIR / 'problem2_diagnostics.csv'}")
    print(f"已保存梯度表 -> {OUT_DIR / 'problem2_severity_table.csv'}")
    print(f"已保存抑郁程度编码敏感性 -> {OUT_DIR / 'problem2_severity_encoding_sensitivity.csv'}")

    plot_severity_gradient(df)
    plot_remission_forest(res_df)
    plot_complaint_burden(res_df)
    print(f"已保存图 -> {OUT_DIR}")

    sig = res_df[res_df["显著_FDR"]]
    print(f"\n=== 全部 86 项统一 FDR 校正后显著因素（p_FDR<{ALPHA}，共 {len(sig)} 项）===")
    if not sig.empty:
        show = sig.assign(效应量=[r[r["效应类型"]] for _, r in sig.iterrows()])
        print(show[["结局", "组别", "变量", "效应类型", "效应量", "CI下", "CI上", "p", "p_FDR"]]
              .to_string(index=False))
    sig_within = res_df[res_df["显著_模型内FDR"]]
    print(f"\n模型内 FDR 敏感性口径下显著因素：{len(sig_within)} 项；"
          "该口径仅作对照，不替代全局主口径。")


if __name__ == "__main__":
    main()
