"""口径不变量的回归检验：ADR 里写死为"事实"的数字，代码不天然保证，靠这里的断言防止悄悄漂移。

不用 pytest，直接跑：.venv/bin/python code/test_definitions.py
"""

import numpy as np
import pandas as pd

import common as c

df, meta = c.build_cohort()

# ADR-0010：附件2 重复录入按并集去重后，队列恢复与附件1 的 1:1 连接，N=3149。
assert meta["建模队列N"] == 3149, f"队列 N={meta['建模队列N']}，应为 3149"

# ADR-0003：df["缓解"] 必须是单点判据 D12=0，不可悄悄换回两点 D6=D12=0——
# 后者恰是「豁免后持续不适」(D6=1 或 D12=1) 的逐个体补集，会在综合层重复计数同一根轴。
assert (df["缓解"] == (df["D12"] == 0)).all(), "df['缓解'] 不是单点判据 D12=0，ADR-0003 口径被改动"

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

print("全部通过。")
