"""口径不变量的回归检验：ADR 里写死为"事实"的数字，代码不天然保证，靠这里的断言防止悄悄漂移。

不用 pytest，直接跑：.venv/bin/python code/test_definitions.py
"""

import numpy as np
import pandas as pd

import common as c
import problem3_efficacy as q3

df, meta = c.build_cohort()

# ADR-0010：附件2 重复录入按并集去重后，队列恢复与附件1 的 1:1 连接，N=3149。
assert meta["建模队列N"] == 3149, f"队列 N={meta['建模队列N']}，应为 3149"

# ADR-0003：主口径固定为 D12=0，保证输出可复现；面向读者时称“第12月无主诉率（代理缓解率）”。
assert (df["缓解"] == (df["D12"] == 0)).all(), "df['缓解'] 不是单点判据 D12=0，ADR-0003 口径被改动"

# 两点口径是定义敏感性分析。若将它放入综合层，就必须识别它与下列指标互补、避免重复计数。
two_pt_remission = ((df["D6"] == 0) & (df["D12"] == 0)).astype(int)
assert (two_pt_remission == 1 - df["豁免后持续不适"]).all(), "两点缓解与豁免后持续不适不是逐个体互补"

# ADR-0003：二者的 Pearson r 应为 -1.0000（互补变量的定义性结果）。
r = np.corrcoef(two_pt_remission, df["豁免后持续不适"])[0, 1]
assert abs(r - (-1.0)) < 1e-6, f"r={r:.4f}，应为 -1.0000"

# ADR-0003：分组两点缓解率 + 豁免后持续不适率 应在每组都等于 100.00%（互补的另一种表述）。
for g in c.GROUPS:
    sub = df[df["组别"] == g]
    total = two_pt_remission[sub.index].mean() * 100 + sub["豁免后持续不适"].mean() * 100
    assert abs(total - 100.0) < 0.01, f"组{g} 两率之和={total:.2f}，应为 100.00%"

# ADR-0005：附件2 现成的"是否出现不适症状"栏 == 7 类实质症状(SUBST，不含失访)在该时点标记数之和，
# 4 个时点零误差；重新逐行读取附件2 原始表核验（build_cohort 的去重环节会丢弃该栏）。
names = ["序号", "组别"] + [f"{s}@{t}" for s in c.SYMPTOMS + ["是否出现不适症状"] for t in c.TIMES]
frames = []
for sheet, hosp in [("一院随访的抗抑郁药物使用后主诉情况", "一院"),
                    ("二院随访的抗抑郁药物使用后主诉情况", "二院")]:
    d = pd.read_excel(c.F2, sheet_name=sheet, header=None, skiprows=3, names=names).dropna(how="all")
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
for t in c.TIMES:
    lhs = raw[f"是否出现不适症状@{t}"].fillna(0)
    rhs = raw[[f"{s}@{t}" for s in c.SUBST]].fillna(0).sum(axis=1)
    assert (lhs == rhs).all(), f"第{t}月：现成栏与7症状之和不一致，共 {(lhs != rhs).sum()} 处"

# ADR-0013：探索性子组同时保留较窄和较宽定义，且窄定义必须是宽定义的子集。
q3df, _ = q3.build_q3_cohort()
early_d1 = q3.adaptation_cohort(q3df)
early_d1_or_d3 = q3.adaptation_cohort(q3df, include_month3=True)
assert len(early_d1) == 835, f"D1=1 子组 n={len(early_d1)}，应为 835"
assert len(early_d1_or_d3) == 969, f"D1=1或D3=1 子组 n={len(early_d1_or_d3)}，应为 969"
assert set(early_d1.index) <= set(early_d1_or_d3.index), "D1=1 子组不是宽定义子组的子集"

# ADR-0012：受试者 bootstrap 应返回三药且排名第一概率总和约为 100%。
_, boot = q3.bootstrap_topsis(q3df, n_draw=100, seed=20260717)
rank_rows = boot[boot["区块"] == "药物排名"]
assert set(rank_rows["比较"]) == {"A", "B", "C"}, "bootstrap 排名结果缺少药物"
rank_sum = rank_rows["排名第一概率%"].astype(float).sum()
assert abs(rank_sum - 100.0) <= 0.2, f"bootstrap 排名概率和={rank_sum}%，应约为100%"

print("全部通过。")
