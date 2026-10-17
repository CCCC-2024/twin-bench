# analyze_error_vs_decision.py 输出 · 2026-10-06

运行：云端 Linux，Python 3 + numpy 2.5.3 + scipy 1.18.1；脚本、作者 UAV.py / planner.py / UAVconfig.json 未修改。20 个种子 × 60 步。默认 twin 的决策不一致率 0.02417 与 10/2 `mc_summary.json` 的 action_mismatch 一致。

## 原始输出
```
2g separation (Mahalanobis, 24ch) one grid step: comp1 median 1.9  comp2 median 0.2
3g separation (Mahalanobis, 24ch) one grid step: comp1 median 2.3  comp2 median 0.3
2g (60,60)->(60,80) sep 0.5 ; (60,60)->(80,60) sep 3.4
3g (60,60)->(60,80) sep 0.6 ; (60,60)->(80,60) sep 4.2
default twin: {'p_err1_gt10': 0.035, 'p_err2_gt10': 0.30916666666666665, 'p_decision_mismatch': 0.024166666666666666, 'late_err2_mean': 5.412101495876666, 'late_bias2': 1.96617358304092}
twin p x0.2: {'p_err1_gt10': 0.07333333333333333, 'p_err2_gt10': 0.7983333333333333, 'p_decision_mismatch': 0.030833333333333334, 'late_err2_mean': 14.950159487687982, 'late_bias2': -14.296517744182212}
twin p x0.0: {'p_err1_gt10': 0.7333333333333333, 'p_err2_gt10': 0.9, 'p_decision_mismatch': 0.48333333333333334, 'late_err2_mean': 60.00000000000001, 'late_bias2': -60.00000000000001}
mismatch timesteps: [31, 32] count 29
```

## 读法
- 24 个应变通道对 component 2 的损伤几乎分辨不出（相邻损伤档的马氏距离中位数 0.2–0.3，component 1 为 1.9–2.3）。
- 默认 twin：component 2 估计偏差 >10 个损伤百分点的时间步占 31%，component 1 占 3.5%；所选动作与“按真实状态查策略”的动作不同的时间步占 2.4%，全部出现在第 31–32 步（component 1 在第 30 步跳档后滞后 1–2 步）。
- twin 损伤增长概率 ×0.2：component 2 偏差 >10 点占 80%，动作不一致仅 3.1%。×0（twin 认为损伤不增长）：动作不一致 48%。
- 范围：作者的合成资产（脚本化损伤路径，动作不影响损伤增长，见 nodes/asset.py `custom`）；参考动作为最近网格真值状态下的策略；“偏差 >10 点”的阈值是本脚本设定。不代表真实 UAV 或一般 DT 的误差率。
