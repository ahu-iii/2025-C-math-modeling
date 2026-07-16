"""
问题3：三种抗抑郁药物的疗效综合评价。

四层结构（前三层沿用 ADR-0007，综合层按 ADR-0012 改口径，适应期转归层为 ADR-0013 新增）：

  1. 指标层   逐指标组间比较（缓解率/复发率/豁免后持续不适率/自杀倾向率/ΔB/…）+ Wilson CI + BH 校正
  2. 轨迹层   GEE：P(D_t=1) ~ 药物 + log(时间) + 药物×log(时间)，可交换相关 —— 回答"用药前后变化"
  3. 适应期转归层  限定 D_1=1 子群的后续转归 + KM/Log-rank —— ADR-0002 适应期规则的正面操作化
  4. 综合层   TOPSIS：等权为主口径、熵权为敏感性对照 + Dirichlet 单纯形全空间扫描

口径要点（改动前务必先读对应 ADR）：
  · 结局一律基于"不适主诉"而非"抑郁症状" —— 数据中无随访期抑郁测量，代理假设见 ADR-0014。
  · D_t / B_t 定义、失访不计入：ADR-0001、ADR-0005。附件2 去重取并集：ADR-0010。
  · 缓解 = D12=0（单点，open-Q1 默认，未定案）；曾缓解 = D3=0 或 D6=0（open-Q2 默认，未定案）；
    复发率分母 = 曾缓解人数（open-Q3 默认，未定案）。三者若改口径，综合层需整体重跑。
  · 综合层指标集剪枝为 4 项：ΔB 因基线依赖出局（C 的 ΔB 最高仅因 C 的 B_1 最高）、
    末次负担 B12 因与缓解率同轴重复计数出局、停药率因与持续不适率重叠出局。详见 ADR-0012。
  · 熵权法在 3 备选下失效，不作主口径，仅作对照并在论文中报告其失效。失效的证据不是"权重看着
    不合理"，而是**熵权的两种通行预处理自相矛盾**：同一指标「缓解率」在原始值口径下权重 0.0011、
    在极差归一口径下 0.4532，相差 423 倍，且 A/C 次序在二者间翻转。故两种口径都实现、都报告。

输出（论文插图与诊断图分级见 ADR-0011）：
  output/problem3_indicators.csv   - 指标层：各药各指标 率/CI + 组间检验 + BH 校正
  output/problem3_gee.csv          - 轨迹层：GEE 系数/OR/CI/p + 交互项联合检验
  output/problem3_adaptation.csv   - 适应期转归层：D_1=1 子群各时点转归 + Log-rank
  output/problem3_topsis.csv       - 综合层：指标矩阵、三套权重(等权/熵权×2)、贴近度、
                                     熵权预处理分歧倍数、留一稳健性、全空间扫描
  output/problem3_trajectory.png            - 图8：全队列三药不适率轨迹（观测 + GEE 拟合）
  output/problem3_adaptation_trajectory.png - 图9：D_1=1 子群的后续转归（与图8 构成对照）
  output/problem3_weight_sensitivity.png    - 图10：权重全空间下 TOPSIS 贴近度的分布
  output/diagnostics/problem3_gee_fit_check.png  - 诊断：GEE 拟合 vs 观测逐格核对
  output/diagnostics/problem3_adaptation_km.png  - 诊断：KM 曲线（退化成 3 级阶梯，不入论文）
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 复用问题2 的队列构造：附件1 左连接附件2、附件2 按并集去重、D_t/B_t 构造、中文字体。
# 三问共用同一份口径，避免各自实现导致缓解率对不上（这正是 ADR-0001 设立的初衷）。
from problem2_factors import (  # noqa: E402
    OUT_DIR,
    SUBST,
    TIMES,
    build_cohort,
)

warnings.filterwarnings("ignore")

DIAG_DIR = OUT_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
DRUGS = ["C", "A", "B"]                      # 固定展示顺序：对照在前
DRUG_OF_GROUP = {1: "C", 2: "A", 3: "B"}
DRUG_LABEL = {"C": "C（对照药）", "A": "A（新药）", "B": "B（新药）"}
DRUG_COLORS = {"C": "#5C7CA3", "A": "#D97757", "B": "#788C5D"}

# 综合层指标集（ADR-0012 剪枝后）：方向 True=效益型（越大越好）/ False=成本型（越小越好）
TOPSIS_INDICATORS = {
    "缓解率": True,
    "复发率": False,
    "豁免后持续不适率": False,
    "自杀倾向率": False,
}


def month_log_axis(ax, pts):
    """时间用对数横轴时，只保留随访时点这几个刻度。

    matplotlib 的 log 轴默认会自作主张加次刻度（如 4×10⁰），在只有 4 个真实时点的图上
    纯属噪声、还会与主刻度标签打架 —— 与 ADR-0011 记录的问题2 森林图 log 轴缺陷同源。
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(pts))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t} 月" for t in pts])


# --------------------------------------------------------------------------
# 队列：在问题2 的基础上补问题3 专用的派生变量
# --------------------------------------------------------------------------
def build_q3_cohort():
    """问题2 队列 + 问题3 派生变量。返回 (df, meta)。"""
    df, meta = build_cohort()
    df["药"] = df["组别"].map(DRUG_OF_GROUP)

    # 适应期（ADR-0002）：前期(1或3月)有不适，但后期(6且12月)均无 → 视为已适应，前三月不计入药物质量问题
    df["适应期"] = (((df.D1 == 1) | (df.D3 == 1)) & (df.D6 == 0) & (df.D12 == 0)).astype(int)
    # 豁免后持续不适：把适应期个体豁免掉之后，仍构成"药物质量问题"的人 —— 即后期(6或12月)仍有不适
    df["豁免后持续不适"] = ((df.D6 == 1) | (df.D12 == 1)).astype(int)

    df["缓解"] = (df.D12 == 0).astype(int)                    # ADR-0003 单点默认（open-Q1 未定案）
    df["曾缓解"] = ((df.D3 == 0) | (df.D6 == 0)).astype(int)  # ADR-0004 宽松默认（open-Q2 未定案）
    df["复发"] = ((df["曾缓解"] == 1) & (df.D12 == 1)).astype(int)
    df["任何主诉"] = df[[f"D{t}" for t in TIMES]].max(axis=1)
    df["ΔB"] = df["B1"] - df["B12"]                           # 描述性；不进综合层（ADR-0012）

    for s in SUBST + ["失访"]:                                # 各症状"任一时点是否出现"
        df[f"任一_{s}"] = df[[f"{s}@{t}" for t in TIMES]].fillna(0).gt(0).max(axis=1).astype(int)
    return df, meta


# --------------------------------------------------------------------------
# 第 1 层：指标层
# --------------------------------------------------------------------------
# (指标名, 分子列, 分母限定列或 None, 是否进综合层)
RATE_SPECS = [
    ("缓解率", "缓解", None),
    ("复发率", "复发", "曾缓解"),
    ("豁免后持续不适率", "豁免后持续不适", None),
    ("自杀倾向率", "任一_有自杀倾向", None),
    ("任何主诉率", "任何主诉", None),
    ("适应期比例", "适应期", None),
    ("停药率", "任一_副作用导致停药", None),
    ("失访率", "任一_失访", None),
]


def rate_table(df):
    """各药各指标的率 + Wilson 95%CI + 三组卡方；BH 校正跨指标。"""
    rows, tests = [], []
    for name, num, denom in RATE_SPECS:
        tab = []
        for g in DRUGS:
            sub = df[df.药 == g]
            sub = sub if denom is None else sub[sub[denom] == 1]
            k, n = int(sub[num].sum()), len(sub)
            lo, hi = proportion_confint(k, n, method="wilson")
            rows.append({"指标": name, "药": g, "分母口径": denom or "全体",
                         "事件数": k, "n": n, "率%": round(100 * k / n, 2),
                         "Wilson CI下%": round(100 * lo, 2), "Wilson CI上%": round(100 * hi, 2)})
            tab.append([k, n - k])
        chi2, p, _, _ = stats.chi2_contingency(np.array(tab))
        tests.append({"指标": name, "检验": "Pearson 卡方(3组)",
                      "统计量": round(chi2, 3), "p": round(p, 4)})

    # ΔB：连续/计数型，用 Kruskal-Wallis（ADR-0012：仅描述，不进综合层）
    kw = stats.kruskal(*[df[df.药 == g]["ΔB"] for g in DRUGS])
    for g in DRUGS:
        s = df[df.药 == g]
        rows.append({"指标": "ΔB(=B1-B12)均值", "药": g, "分母口径": "全体",
                     "事件数": "", "n": len(s), "率%": round(s["ΔB"].mean(), 3),
                     "Wilson CI下%": "", "Wilson CI上%": ""})
    tests.append({"指标": "ΔB(=B1-B12)均值", "检验": "Kruskal-Wallis",
                  "统计量": round(kw.statistic, 3), "p": round(kw.pvalue, 4)})

    tests = pd.DataFrame(tests)
    # BH 校正，与问题1/问题2 及 建模方案§七 口径一致；校正族=指标层的全部组间检验
    tests["p_BH"] = multipletests(tests["p"], method="fdr_bh")[1].round(4)
    tests["显著_BH"] = tests["p_BH"] < ALPHA
    return pd.DataFrame(rows), tests


# --------------------------------------------------------------------------
# 第 2 层：轨迹层（GEE）
# --------------------------------------------------------------------------
def to_long(df):
    """宽表 → 长表（每人 4 行），供 GEE 使用。"""
    long = df.melt(id_vars=["药"], value_vars=[f"D{t}" for t in TIMES],
                   var_name="tvar", value_name="D", ignore_index=False)
    long["pid"] = long.index
    long["月"] = long["tvar"].str[1:].astype(int)
    # log(月) 而非月：不适率随时间近似指数衰减，log 尺度下轨迹接近直线，交互项才是"下降速度之比"
    long["logt"] = np.log(long["月"])
    return long.sort_values(["pid", "月"]).reset_index(drop=True)


def _joint_wald(res, pattern):
    """对系数名含 pattern 的全部项做联合 Wald 检验。"""
    names = list(res.params.index)
    idx = [i for i, n in enumerate(names) if pattern in n]
    R = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1
    w = res.wald_test(R, scalar=False)
    return float(np.ravel(w.statistic)[0]), float(w.pvalue), len(idx)


def fit_gee(long):
    """GEE 主模型（含交互）+ 无交互模型；返回 (系数表, 检验表, 主模型)。"""
    fml = "D ~ C(药, Treatment('C')) * logt"
    m = smf.gee(fml, "pid", data=long, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    m0 = smf.gee("D ~ C(药, Treatment('C')) + logt", "pid", data=long,
                 family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()

    rows = []
    for res, tag in [(m, "含交互(主模型)"), (m0, "无交互")]:
        ci = res.conf_int()
        for nm in res.params.index:
            rows.append({"模型": tag, "项": nm,
                         "系数": round(res.params[nm], 4), "SE": round(res.bse[nm], 4),
                         "OR": round(np.exp(res.params[nm]), 3),
                         "OR CI下": round(np.exp(ci.loc[nm, 0]), 3),
                         "OR CI上": round(np.exp(ci.loc[nm, 1]), 3),
                         "p": round(res.pvalues[nm], 4)})

    chi2, p, dfree = _joint_wald(m, ":")
    tests = [{"检验": "药物×log(时间) 交互 联合Wald", "df": dfree,
              "统计量": round(chi2, 3), "p": round(p, 4),
              "结论": "未观测到显著差异" if p >= ALPHA else "存在显著差异"}]
    chi2b, pb, dfb = _joint_wald(m0, "T.")
    tests.append({"检验": "药物主效应 联合Wald(无交互模型)", "df": dfb,
                  "统计量": round(chi2b, 3), "p": round(pb, 4),
                  "结论": "未观测到显著差异" if pb >= ALPHA else "存在显著差异"})

    # 时间为分类的敏感性版本：不假定 log 线性，交互项 6 df
    mc = smf.gee("D ~ C(药, Treatment('C')) * C(月)", "pid", data=long,
                 family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
    chi2c, pc, dfc = _joint_wald(mc, ":")
    tests.append({"检验": "药物×时间 交互 联合Wald（时间为分类，敏感性）", "df": dfc,
                  "统计量": round(chi2c, 3), "p": round(pc, 4),
                  "结论": "未观测到显著差异" if pc >= ALPHA else "存在显著差异"})
    tests.append({"检验": "可交换相关参数 ρ", "df": "",
                  "统计量": round(m.cov_struct.dep_params, 4), "p": "",
                  "结论": "同一受试者 4 次观测的组内相关"})
    return pd.DataFrame(rows), pd.DataFrame(tests), m


# --------------------------------------------------------------------------
# 第 3 层：适应期转归层（ADR-0013）
# --------------------------------------------------------------------------
def adaptation_cohort(df):
    """限定第 1 月已出现不适者（D_1=1）—— 即 ADR-0002 适应期规则所指的人群。"""
    return df[df.D1 == 1].copy()


def survival_vars(sub):
    """首次不适消退的 (时间, 事件) 。

    事件 = 首个 D_t=0 的月份（t=3/6/12）；至 12 月仍未消退 → 在 12 月右删失。
    失访单独处理：ADR-0001 规定失访不计入 D_t，故失访者该时点 D_t=0，若直接采信会被
    误判为"已消退"。生存分析自带删失机制，故在此按"首次失访时点右删失"处理 —— 这比
    全队列指标层的口径更严格，且不与 ADR-0001 冲突（ADR-0001 管的是 D_t 怎么定义，
    这里管的是该观测算不算数）。
    """
    T, E = [], []
    for _, r in sub.iterrows():
        t_end, ev = 12, 0
        for t in [3, 6, 12]:
            if r[f"失访@{t}"] == 1:               # 失访 → 就此删失，不再采信其后的 D_t
                t_end, ev = t, 0
                break
            if r[f"D{t}"] == 0:                   # 首次消退
                t_end, ev = t, 1
                break
        T.append(t_end)
        E.append(ev)
    return np.array(T), np.array(E)


def adaptation_table(sub):
    """D_1=1 子群：各时点仍不适率 + 组间卡方；始终未消退者比例；Log-rank。"""
    rows, tests = [], []
    for t in [3, 6, 12]:
        tab = []
        for g in DRUGS:
            s = sub[sub.药 == g]
            k, n = int(s[f"D{t}"].sum()), len(s)
            lo, hi = proportion_confint(k, n, method="wilson")
            rows.append({"指标": f"第{t}月仍不适率", "药": g, "事件数": k, "n": n,
                         "率%": round(100 * k / n, 2),
                         "Wilson CI下%": round(100 * lo, 2), "Wilson CI上%": round(100 * hi, 2)})
            tab.append([k, n - k])
        chi2, p, _, _ = stats.chi2_contingency(np.array(tab))
        tests.append({"检验": f"第{t}月仍不适率 卡方(3组)", "统计量": round(chi2, 3),
                      "p": round(p, 4)})

    never = ((sub.D3 == 1) & (sub.D6 == 1) & (sub.D12 == 1)).astype(int)
    tab = []
    for g in DRUGS:
        m = sub.药 == g
        k, n = int(never[m].sum()), int(m.sum())
        lo, hi = proportion_confint(k, n, method="wilson")
        rows.append({"指标": "始终未消退率(D3=D6=D12=1)", "药": g, "事件数": k, "n": n,
                     "率%": round(100 * k / n, 2),
                     "Wilson CI下%": round(100 * lo, 2), "Wilson CI上%": round(100 * hi, 2)})
        tab.append([k, n - k])
    chi2, p, _, _ = stats.chi2_contingency(np.array(tab))
    tests.append({"检验": "始终未消退率 卡方(3组)", "统计量": round(chi2, 3), "p": round(p, 4)})

    T, E = survival_vars(sub)
    lr = multivariate_logrank_test(T, sub["药"].values, E)
    tests.append({"检验": "首次消退 Log-rank(3组)", "统计量": round(lr.test_statistic, 3),
                  "p": round(lr.p_value, 4)})

    tests = pd.DataFrame(tests)
    tests["p_BH"] = multipletests(tests["p"], method="fdr_bh")[1].round(4)
    tests["显著_BH"] = tests["p_BH"] < ALPHA
    return pd.DataFrame(rows), tests


# --------------------------------------------------------------------------
# 第 4 层：综合层（TOPSIS，ADR-0012）
# --------------------------------------------------------------------------
def indicator_matrix(df):
    """综合层的 3×4 指标矩阵（比例，非百分数）。"""
    spec = {name: (num, denom) for name, num, denom in RATE_SPECS}
    M = {}
    for g in DRUGS:
        row = {}
        for name in TOPSIS_INDICATORS:
            num, denom = spec[name]
            sub = df[df.药 == g]
            sub = sub if denom is None else sub[sub[denom] == 1]
            row[name] = sub[num].mean()
        M[g] = row
    return pd.DataFrame(M).T[list(TOPSIS_INDICATORS)]


def topsis(M, W):
    """极差归一化(方向统一) → 加权 → 与正/负理想解的欧氏距离 → 贴近度 C_i=D-/(D++D-)。"""
    X = M.copy()
    for c in X.columns:
        v = X[c].astype(float)
        rng = v.max() - v.min()
        if rng == 0:                                    # 全等则该列不含区分信息，置 0 而非除零
            X[c] = 0.0
        else:
            X[c] = (v - v.min()) / rng if TOPSIS_INDICATORS[c] else (v.max() - v) / rng
    V = X.values * np.asarray(W, float)
    Dp = np.sqrt(((V - V.max(0)) ** 2).sum(1))          # 到正理想解
    Dm = np.sqrt(((V - V.min(0)) ** 2).sum(1))          # 到负理想解
    with np.errstate(invalid="ignore"):
        C = np.where(Dp + Dm > 0, Dm / (Dp + Dm), 0.5)
    return pd.Series(C, index=M.index)


def _entropy_from(R):
    """由（已按各自口径预处理好的）矩阵算熵权：按列和归一 → 信息熵 → 差异系数 → 归一。"""
    R = np.asarray(R, float)
    P = R / R.sum(axis=0)                                # 按列归一（不是全局）
    k = 1 / np.log(R.shape[0])
    E = -k * np.nansum(np.where(P > 0, P * np.log(P), 0), axis=0)
    d = 1 - E                                            # 差异系数
    return d / d.sum()


def entropy_weights(M, preprocess="raw"):
    """熵权法。保留为敏感性对照 —— 本题 3 备选下失效，不作主口径（ADR-0012）。

    熵权法在预处理上有两种通行写法，本函数都实现，因为**二者的分歧本身就是主要证据**：

    · preprocess="raw"    直接对原始值按列和归一。熵度量各指标的相对离散度（近似变异系数），
                          方向不影响熵（熵measure的是离散度，"越大越好/越小越好"不改变离散度，
                          方向由 TOPSIS 的正向化负责）。
    · preprocess="minmax" 先方向统一+极差归一到 [0,1] 再和归一（教科书常见写法）。极差归一
                          会把每列的 min 压成 0（ln 0 未定义），故按惯例做 eps 平移。

    两者在本题给出的权重相差 423 倍（缓解率 0.0011 vs 0.4532），且 A/C 次序翻转 —— 见
    ADR-0012「结果」。原因：n=3 时极差归一必然把 [min, mid, max] 拉伸到 [0, x, 1] 撑满全幅，
    无论原始差异是 2 个百分点还是纯噪声；而原始值口径下 94.66/94.67/96.48 的相对离散度趋近 0。
    同一病根（3 个备选估不出有意义的离散度）的两种相反伪影。
    """
    if preprocess == "raw":
        return _entropy_from(M.values)
    if preprocess == "minmax":
        X = M.copy()
        for c in X.columns:
            v = X[c].astype(float)
            rng = v.max() - v.min()
            if rng == 0:
                X[c] = 0.5
            else:
                X[c] = (v - v.min()) / rng if TOPSIS_INDICATORS[c] else (v.max() - v) / rng
        eps = 1e-3                                       # 平移出 0，使 ln 有定义
        return _entropy_from(X.values * (1 - 2 * eps) + eps)
    raise ValueError(f"未知的 preprocess: {preprocess}")


def weight_space_scan(M, n_draw=20000, seed=0):
    """Dirichlet 单纯形全空间扫描：ADR-0007 要求的权重敏感性分析的具体化（ADR-0012）。

    在权重单纯形上均匀采样，统计各药排名第一的比例与贴近度分布 —— 回答"排名是否
    依赖权重的选取"，且不依赖任选的若干套方案。返回 (贴近度矩阵 n_draw×3, 第一名比例)。
    """
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(M.shape[1]), n_draw)
    C = np.array([topsis(M, w).values for w in W])
    first = pd.Series(M.index.values[C.argmax(1)]).value_counts(normalize=True)
    return C, first.reindex(M.index).fillna(0.0)


def topsis_report(M):
    """综合层全部结果整理成长表。"""
    W_eq = np.ones(M.shape[1]) / M.shape[1]
    W_en = entropy_weights(M, "raw")
    W_mm = entropy_weights(M, "minmax")
    C_eq, C_en, C_mm = topsis(M, W_eq), topsis(M, W_en), topsis(M, W_mm)
    C_scan, first = weight_space_scan(M)

    rows = []
    for g in DRUGS:
        for name in M.columns:
            rows.append({"区块": "指标矩阵", "药": g, "项": name,
                         "值": round(M.loc[g, name], 5),
                         "备注": "效益型" if TOPSIS_INDICATORS[name] else "成本型"})
    for name, w_eq, w_en, w_mm in zip(M.columns, W_eq, W_en, W_mm):
        rows.append({"区块": "权重", "药": "", "项": name, "值": round(w_eq, 4),
                     "备注": "等权（主口径）"})
        rows.append({"区块": "权重", "药": "", "项": name, "值": round(w_en, 4),
                     "备注": "熵权-原始值口径（对照，本题失效）"})
        rows.append({"区块": "权重", "药": "", "项": name, "值": round(w_mm, 4),
                     "备注": "熵权-极差归一口径（对照，本题失效）"})
    for g in DRUGS:
        for tag, C in [("等权（主口径）", C_eq), ("熵权-原始值口径", C_en),
                       ("熵权-极差归一口径", C_mm)]:
            rows.append({"区块": "贴近度", "药": g, "项": tag, "值": round(C[g], 4),
                         "备注": f"排名 {int(C.rank(ascending=False)[g])}"})
        rows.append({"区块": "权重全空间扫描", "药": g, "项": "排名第一的比例%",
                     "值": round(100 * first[g], 2), "备注": "Dirichlet 均匀采样 20000 组权重"})
    # 熵权两种预处理的分歧倍数：这是"熵权在 n=3 下无信号"的最直接证据，单独记一行备查
    for name, w_en, w_mm in zip(M.columns, W_en, W_mm):
        rows.append({"区块": "熵权预处理分歧", "药": "", "项": name,
                     "值": round(max(w_en, w_mm) / max(min(w_en, w_mm), 1e-12), 1),
                     "备注": f"原始值 {w_en:.4f} vs 极差归一 {w_mm:.4f} 的倍数"})

    # 留一稳健性：逐个删指标，看排序是否翻转
    for c in M.columns:
        sub = M.drop(columns=[c])
        w = np.ones(sub.shape[1]) / sub.shape[1]
        order = " > ".join(topsis(sub, w).sort_values(ascending=False).index)
        rows.append({"区块": "留一稳健性(等权)", "药": "", "项": f"删除「{c}」后排序",
                     "值": "", "备注": order})
    return pd.DataFrame(rows), C_eq, C_en, C_mm, C_scan, first, W_eq, W_en, W_mm


# --------------------------------------------------------------------------
# 图8：轨迹（论文插图）
# --------------------------------------------------------------------------
def plot_trajectory(df, gee_model):
    """三药不适率随时间的轨迹：观测点 + GEE 拟合曲线。

    承载的论证（ADR-0011 三问）：正文"三药不适率均随时间大幅单调下降，三条轨迹平行、
    置信区间自始至终相互重叠"这一句。图给的是**形状**——衰减的陡峭程度，以及三条线
    平行下坠、误差棒始终交叠（"平行"= 无交互、"交叠"= 无主效应，两个否定结论都是形态，
    图比两个 p 值更一望即知）。这是表格给不了的。

    注意措辞：三条线并非"重合"（C 全程略高、B 全程略低，次序一致），而是**平行且
    区间重叠**。把可见的系统性次序说成"重合"是把图和结论对不上，反为失分点。
    """
    fig, ax = plt.subplots(figsize=(7.6, 5))
    grid = np.linspace(np.log(1), np.log(12), 100)
    for g in DRUGS:
        sub = df[df.药 == g]
        ys, los, his = [], [], []
        for t in TIMES:
            k, n = int(sub[f"D{t}"].sum()), len(sub)
            lo, hi = proportion_confint(k, n, method="wilson")
            ys.append(100 * k / n)
            los.append(100 * (k / n - lo))
            his.append(100 * (hi - k / n))
        ax.errorbar(TIMES, ys, yerr=[los, his], fmt="o", ms=6, capsize=3, elinewidth=1.2,
                    color=DRUG_COLORS[g], label=DRUG_LABEL[g], zorder=3)
        pred = gee_model.predict(pd.DataFrame({"药": g, "logt": grid}))
        ax.plot(np.exp(grid), 100 * np.asarray(pred), "-", lw=1.6,
                color=DRUG_COLORS[g], alpha=0.75, zorder=2)

    month_log_axis(ax, TIMES)
    ax.set_xlabel("随访时点（对数坐标）", fontsize=9)
    ax.set_ylabel("该时点出现不适主诉的比例 / %", fontsize=9)
    ax.set_title("三种药物的不适主诉率随时间的变化\n点=观测率(Wilson 95%CI)；线=GEE 拟合", fontsize=10)
    ax.set_ylim(0, 32)
    ax.grid(axis="y", color="#D1CFC5", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.015, "三条轨迹平行下降、误差棒全程重叠：改善由时间驱动，药物间差异未达显著",
             ha="center", fontsize=7, color="#87867F")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "problem3_trajectory.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图9：适应期转归（论文插图）
# --------------------------------------------------------------------------
def plot_adaptation_trajectory(sub, logrank_p, p12):
    """D_1=1 子群的仍不适率轨迹。

    承载的论证：正文"信号只存在于早期出现不适者这一子群——在这里 A 的转归显著最差"
    这一句。图给的是**形状**：与图8 三条平行下坠的轨迹相对照，本图三条线在第 6→12 月
    **扇形张开**，A 明显掉队。同一批人、同一个时间轴，条件之后信号才现身——这个对照
    是"为什么必须分层看"的视觉论据，表格给不出（表只能给 9 个率）。

    为什么不画 KM 曲线：本子群只有 3 个可能的事件时点，KM 曲线退化成 3 级阶梯，
    其取值恰好就是适应期转归表里已列出的那 9 个数值——按 ADR-0011 第三问（"是否给出
    表格给不了的东西"）不合格，故 KM 图降级为诊断图，Log-rank 检验值保留在表与图题中。
    """
    fig, ax = plt.subplots(figsize=(7.6, 5))
    # 不画第 1 月：该子群按定义在第 1 月恒为 100%，是定义常数而非数据，画上去只会把
    # 真正有信息的 8–44% 区间压扁到图底。起点相同这一事实写在图题里即可。
    pts = [3, 6, 12]
    for g in DRUGS:
        s = sub[sub.药 == g]
        ys, los, his = [], [], []
        for t in pts:
            k, n = int(s[f"D{t}"].sum()), len(s)
            lo, hi = proportion_confint(k, n, method="wilson")
            p = k / n
            ys.append(100 * p)
            los.append(max(0.0, 100 * (p - lo)))                  # 防浮点误差产生负误差棒
            his.append(max(0.0, 100 * (hi - p)))
        ax.errorbar(pts, ys, yerr=[los, his], fmt="o-", ms=6, lw=1.8, capsize=3,
                    elinewidth=1.2, color=DRUG_COLORS[g], label=DRUG_LABEL[g], zorder=3)

    ax.axvspan(6, 12, color="#F0EEE6", zorder=0)                  # 高亮信号出现的区间
    ax.annotate(f"三线在此扇形张开\nA 掉队（第 12 月 χ² p={p12:.4f}）",
                xy=(8.5, 34), fontsize=7.5, color="#87867F", ha="center")
    month_log_axis(ax, pts)
    ax.set_xlabel("随访时点（对数坐标）", fontsize=9)
    ax.set_ylabel("该时点仍有不适的比例 / %", fontsize=9)
    ax.set_title("第 1 月即出现不适者（D₁=1，n=835）的后续转归\n"
                 "三组按定义均自第 1 月的 100% 出发；"
                 f"首次消退 Log-rank p = {logrank_p:.4f}", fontsize=10)
    ax.set_ylim(0, 50)
    ax.grid(axis="y", color="#D1CFC5", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.015, "与图8（全队列三线平行）对照：条件于「早期已出现不适」后，药物差异才显现",
             ha="center", fontsize=7, color="#87867F")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "problem3_adaptation_trajectory.png", dpi=200)
    plt.close(fig)


def plot_adaptation_km_diag(sub, logrank_p):
    """诊断图：D_1=1 子群首次消退的 KM 曲线（不入论文，见 plot_adaptation_trajectory 的说明）。

    留作自查：核对 Log-rank 的删失处理是否如预期（失访者在其失访时点离开风险集）。
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    T, E = survival_vars(sub)
    kmf = KaplanMeierFitter()
    for g in DRUGS:
        m = (sub["药"] == g).values
        kmf.fit(T[m], E[m], label=f"{g} (n={m.sum()}, 事件={int(E[m].sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=DRUG_COLORS[g], lw=1.6,
                                   ci_alpha=0.12)
    ax.set_xlabel("月")
    ax.set_ylabel("不适仍未消退的比例（KM）")
    ax.set_title(f"诊断：D₁=1 子群首次消退 KM\nLog-rank p={logrank_p:.4f}；"
                 "第 12 月的陡降=末次观测的事件与删失同时发生", fontsize=9)
    ax.set_xticks([0, 3, 6, 12])
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "problem3_adaptation_km.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图10：权重敏感性（论文插图）
# --------------------------------------------------------------------------
def plot_weight_sensitivity(C_scan, C_eq, C_en, C_mm, first, index):
    """权重全空间下 TOPSIS 贴近度的分布。

    承载的论证：正文"B 的第一名对权重稳健，而 A 与 C 的次序随权重方案翻转、不可区分"
    这一句。图给的是**形状**——B 的分布几乎整条压在 A/C 之上，而 A 与 C 的分布大面积重叠
    （重叠 = 不可区分，这正是结论本身）。表格只能给"B 第一占 99.9%"这一个数字，给不出
    "A 与 C 重叠到什么程度"。

    措辞纪律：不可写"B 与 A/C **完全**分离"——A 的分布有一条细尾伸到 1.0，正对应扫描里
    A 仍有 0.1% 的权重组合排第一。图上看得见的细尾与那 0.1% 必须对得上，否则图注自相矛盾。
    """
    fig, ax = plt.subplots(figsize=(7.6, 5))
    order = list(index)
    parts = ax.violinplot([C_scan[:, i] for i in range(len(order))],
                          positions=range(len(order)), vert=False,
                          showextrema=False, widths=0.85)
    for pc, g in zip(parts["bodies"], order):
        pc.set_facecolor(DRUG_COLORS[g])
        pc.set_alpha(0.35)
        pc.set_edgecolor(DRUG_COLORS[g])
        pc.set_linewidth(1.2)
    for i, g in enumerate(order):
        ax.plot(C_eq[g], i, "o", ms=8, color=DRUG_COLORS[g], zorder=4)
        ax.plot(C_en[g], i, "D", ms=6, color=DRUG_COLORS[g], mfc="white", mew=1.6, zorder=4)
        ax.plot(C_mm[g], i, "s", ms=5.5, color=DRUG_COLORS[g], mfc="white", mew=1.6, zorder=4)
    # 图例句柄单独构造并用中性色：借某一组的颜色当图例会被读成"那是 C 的点"
    handles = [plt.Line2D([], [], color="#141413", marker="o", ls="none", ms=8,
                          label="等权（主口径）"),
               plt.Line2D([], [], color="#141413", marker="D", ls="none", ms=6, mfc="white",
                          mew=1.6, label="熵权-原始值口径"),
               plt.Line2D([], [], color="#141413", marker="s", ls="none", ms=5.5, mfc="white",
                          mew=1.6, label="熵权-极差归一口径")]

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{DRUG_LABEL[g]}\n排名第一 {100 * first[g]:.1f}%" for g in order],
                       fontsize=8)
    ax.set_xlabel("TOPSIS 贴近度 $C_i$（越大越优）", fontsize=9)
    ax.set_title("排名对权重的敏感性：权重单纯形上均匀采样 20000 组权重\n"
                 "小提琴=贴近度的全空间分布；圆点=等权，空心菱形/方块=熵权的两种预处理", fontsize=10)
    ax.set_xlim(-0.03, 1.15)                                     # 右侧留白安放图例，避免压住 B 的分布
    ax.grid(axis="x", color="#D1CFC5", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    # 图例必须带框：无框图例落在坐标区内会被读成数据点（这正是本图初版的缺陷）
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95,
              edgecolor="#D1CFC5")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.015,
             f"B 的分布几乎整条压在 A/C 之上（A 仅在 {100 * first['A']:.1f}% 的权重组合下反超）→ B 第一稳健；"
             "A 与 C 的分布大面积重叠 → 二者不可区分",
             ha="center", fontsize=7, color="#87867F")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "problem3_weight_sensitivity.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 诊断图：GEE 拟合 vs 观测（不入论文，ADR-0011）
# --------------------------------------------------------------------------
def plot_gee_fit_check(df, gee_model):
    """逐格核对 GEE 拟合值与观测率是否吻合 —— 自查 log 线性设定是否成立。"""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for g in DRUGS:
        sub = df[df.药 == g]
        obs = [sub[f"D{t}"].mean() for t in TIMES]
        pred = gee_model.predict(pd.DataFrame({"药": g, "logt": np.log(TIMES)}))
        ax.plot(100 * np.asarray(obs), 100 * np.asarray(pred), "o", ms=7,
                color=DRUG_COLORS[g], label=g)
        for t, o, p in zip(TIMES, obs, pred):
            ax.annotate(f"{g}{t}月", (100 * o, 100 * p), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    lim = [0, 32]
    ax.plot(lim, lim, "--", color="#87867F", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("观测不适率 / %")
    ax.set_ylabel("GEE 拟合不适率 / %")
    ax.set_title("诊断：GEE(log 时间) 拟合 vs 观测\n偏离对角线 = log 线性设定不成立", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "problem3_gee_fit_check.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    df, meta = build_q3_cohort()
    print(f"附件2 原始 {meta['附件2原始行数']} 行 → 去重删除 {meta['去重删除行数']} 行；"
          f"建模队列 N={meta['建模队列N']}（应=3149）")
    print(f"各组 n: {df['药'].value_counts().reindex(DRUGS).to_dict()}")

    # ---- 第 1 层：指标层 ----
    rates, rate_tests = rate_table(df)
    print("\n=== 第1层 指标层：各指标率（%）===")
    print(rates.pivot_table(index="指标", columns="药", values="率%", sort=False)
          .reindex(columns=DRUGS).to_string())
    print("\n=== 指标层 组间检验（BH 校正）===")
    print(rate_tests.to_string(index=False))

    # ---- 第 2 层：轨迹层 ----
    long = to_long(df)
    gee_coef, gee_tests, gee_model = fit_gee(long)
    print("\n=== 第2层 轨迹层：GEE 主模型（含交互）===")
    print(gee_coef[gee_coef["模型"] == "含交互(主模型)"].to_string(index=False))
    print("\n=== 轨迹层 检验 ===")
    print(gee_tests.to_string(index=False))

    # ---- 第 3 层：适应期转归层 ----
    sub = adaptation_cohort(df)
    adapt_rates, adapt_tests = adaptation_table(sub)
    print(f"\n=== 第3层 适应期转归层：D₁=1 子群 n={len(sub)} "
          f"（{sub['药'].value_counts().reindex(DRUGS).to_dict()}）===")
    print(adapt_rates.pivot_table(index="指标", columns="药", values="率%", sort=False)
          .reindex(columns=DRUGS).to_string())
    print("\n=== 适应期转归层 检验（BH 校正）===")
    print(adapt_tests.to_string(index=False))

    # ---- 第 4 层：综合层 ----
    M = indicator_matrix(df)
    topsis_df, C_eq, C_en, C_mm, C_scan, first, W_eq, W_en, W_mm = topsis_report(M)
    print("\n=== 第4层 综合层：指标矩阵（%）===")
    print((M * 100).round(2).to_string())
    print("\n=== 权重（熵权的两种通行预处理并列，其分歧本身是主要证据）===")
    print(pd.DataFrame({"等权(主口径)": W_eq, "熵权-原始值(对照)": W_en.round(4),
                        "熵权-极差归一(对照)": W_mm.round(4)}, index=M.columns).to_string())
    print("\n=== TOPSIS 贴近度 ===")
    print(pd.DataFrame({"等权(主口径)": C_eq.round(4), "熵权-原始值": C_en.round(4),
                        "熵权-极差归一": C_mm.round(4),
                        "全空间第一比例%": (100 * first).round(2)}).to_string())
    print("\n=== 留一稳健性 ===")
    print(topsis_df[topsis_df["区块"] == "留一稳健性(等权)"][["项", "备注"]].to_string(index=False))

    # ---- 输出 ----
    rates_out = pd.concat([rates.assign(层="指标层"),
                           adapt_rates.assign(层="适应期转归层", 分母口径="D₁=1 子群")])
    rates_out.to_csv(OUT_DIR / "problem3_indicators.csv", index=False, encoding="utf-8-sig")
    pd.concat([rate_tests.assign(层="指标层"), adapt_tests.assign(层="适应期转归层")]) \
        .to_csv(OUT_DIR / "problem3_adaptation.csv", index=False, encoding="utf-8-sig")
    pd.concat([gee_coef.assign(区块="系数"), gee_tests.assign(区块="检验")]) \
        .to_csv(OUT_DIR / "problem3_gee.csv", index=False, encoding="utf-8-sig")
    topsis_df.to_csv(OUT_DIR / "problem3_topsis.csv", index=False, encoding="utf-8-sig")

    logrank_p = float(adapt_tests.loc[adapt_tests["检验"].str.contains("Log-rank"), "p"].iloc[0])
    p_d12 = float(adapt_tests.loc[adapt_tests["检验"] == "第12月仍不适率 卡方(3组)", "p"].iloc[0])
    plot_trajectory(df, gee_model)
    plot_adaptation_trajectory(sub, logrank_p, p_d12)
    plot_weight_sensitivity(C_scan, C_eq, C_en, C_mm, first, M.index)
    plot_gee_fit_check(df, gee_model)
    plot_adaptation_km_diag(sub, logrank_p)
    print(f"\n已保存表 -> {OUT_DIR}/problem3_*.csv")
    print(f"已保存论文插图 -> {OUT_DIR}/problem3_*.png（图8/9/10）")
    print(f"已保存诊断图 -> {DIAG_DIR}/problem3_*.png")

    # ---- 结论摘要 ----
    p_int = float(gee_tests.loc[gee_tests["检验"].str.contains("交互 联合Wald$", regex=True), "p"].iloc[0])
    rank_eq = " > ".join(C_eq.sort_values(ascending=False).index)
    rank_en = " > ".join(C_en.sort_values(ascending=False).index)
    rank_mm = " > ".join(C_mm.sort_values(ascending=False).index)
    swing = max(W_mm[0], W_en[0]) / max(min(W_mm[0], W_en[0]), 1e-12)   # 缓解率的权重分歧倍数
    print("\n" + "=" * 72)
    print("结论摘要")
    print("=" * 72)
    print(f"1. 轨迹：时间主效应 OR={np.exp(gee_model.params['logt']):.3f}/log月 "
          f"(p={gee_model.pvalues['logt']:.2e})；药物×时间 交互 p={p_int:.4f} → "
          f"{'未观测到显著差异' if p_int >= ALPHA else '存在显著差异'}")
    # 两套方案的排序在此并列打印，是为了显示 A/C 次序**翻转**（这正是"二者不可区分"的证据），
    # 不是在报告全序。结论口径以下一行为准（ADR-0012 明令不得报告 B>A>C 这样的全序）。
    print(f"2. 综合：各方案排序 —— 等权 {rank_eq}；熵权-原始值 {rank_en}；熵权-极差归一 {rank_mm} "
          f"（A/C 次序在方案间翻转 = 二者不可区分的直接证据）")
    print(f"   熵权失效：同一指标「缓解率」在熵权的两种通行预处理下权重相差 {swing:.0f} 倍 "
          f"（{W_en[0]:.4f} vs {W_mm[0]:.4f}）→ n=3 下熵权无信号可提取，不作主口径")
    print(f"   权重全空间扫描：B 第一占 {100 * first['B']:.1f}%、A 占 {100 * first['A']:.1f}%、"
          f"C 占 {100 * first['C']:.1f}%")
    print("   → 结论口径（ADR-0012）：B 优于 A 和 C（对权重稳健）；A 与 C 不可区分、并列。不报全序。")
    print(f"3. 适应期转归：D₁=1 子群第 12 月仍不适 "
          f"{{{', '.join(f'{g}:{100 * sub[sub.药 == g].D12.mean():.1f}%' for g in DRUGS)}}}，"
          f"卡方 p={p_d12:.4f}")


if __name__ == "__main__":
    main()
