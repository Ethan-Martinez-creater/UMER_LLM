# round_024_full_consistency_stochastic_pair

方案标识：`round_024_full_consistency_stochastic_pair`

## 立项依据

Round21 在训练时随机抽取两个标准视图，分类监督、融合和一致性均只使用该随机视图对；PHEME 五折为 `0.8951/0.8955/0.8890/0.8629`。Round23 改为四视图全部参与分类与融合后，fold2/fold4 改善，但 fold1/fold3/fold5 退化，五折降至 `0.8911/0.8915/0.8846/0.8572`；与此同时 Ma-Weibo 提升至约 `0.9702`。证据表明，小规模 PHEME 需要保留随机视图组合增强，但未被抽中的视图仍可能缺少当步一致性约束。

Round24 每步仍编码四个标准视图，但对每个 event 无放回随机抽取两个视图，仅这两个视图参与文本分类 CE、视图均值和最终融合；广义 Jensen-Shannon 一致性损失单独覆盖全部四个视图。验证和测试仍平均全部四个视图。因此相对 Round21 的唯一机制变化是“一致性覆盖从随机二视图扩大到四视图”，而分类/融合仍保留随机双视图增强；相对 Round23 则恢复随机双视图分类/融合。

模型仍为 clean 初始化、DeBERTa-v3-large、图/文本/检索联合训练、线性融合、平方根逆频率类别权重、单 checkpoint 推理。没有组合多个已训练模型，也没有测试期模型集成。

本方案由 Round21/23 的逐折实验证据直接推导，不新增外部论文引用；已查询 `analysis/literature_reference_ledger.md`，与 Round16 R-Drop、Round22 checkpoint averaging 及 Round23 全四视图分类具有实质差异。

## 强制 fold1+fold2 双折筛选

固定 `partition_seed=3090`、`training_seed=2000`，顺序连续运行 fold1、fold2。两折使用完全相同配置：`train_view_count=4`、`classification_view_count=2`、`view_consistency_weight=0.5`、物理 event batch=4、梯度累积16、等效 event batch=64、最多30 epoch、patience=7。

Round21 同折基线：

| fold | Accuracy | Weighted-F1 | Macro-F1 | Rumor-F1 |
|---|---:|---:|---:|---:|
| 1 | 0.9035 | 0.9038 | 0.8977 | 0.8732 |
| 2 | 0.8809 | 0.8819 | 0.8752 | 0.8484 |
| 二折均值 | 0.8922 | 0.8929 | 0.8864 | 0.8608 |

预声明扩展门槛：

1. fold1 与 fold2 的 Accuracy、Weighted-F1、Macro-F1 任一项相对 Round21 同折下降不得超过 `0.003`；
2. Round24 二折 Accuracy、Weighted-F1、Macro-F1 均值必须全部严格高于 Round21 二折均值；
3. Round24 二折 Macro-F1 均值至少提高 `0.005`；
4. Rumor-F1 完整报告，但不作为本阶段硬门槛；
5. 两折结束后只生成 `twofold_comparison.json`，不得自动运行 fold3--5。只有人工核验门槛通过后，才能以冻结配置补齐其余三折。

输出目录：`results/round_024_full_consistency_stochastic_pair/twofold_seed2000/`。

## 启动记录

2026-07-22 已完成服务器 Shell/Python 语法检查、CLI 参数检查及 10 项单元/前向冒烟测试。随后启动 fold1→fold2 连续双折任务，后台主 PID `1848701`；脚本在两折结束后仅生成 `twofold_comparison.json`，不会自动执行 fold3--5。

## fold1 结果与用户指定停止

fold1 在 epoch5 取得验证选择的最佳单 checkpoint，测试为 Accuracy `0.8708`、Weighted-F1 `0.8720`、Macro-F1 `0.8648`、Rumor-F1 `0.8363`。相对 Round21 fold1 的 `0.9035/0.9038/0.8977/0.8732`，四项分别下降约 `0.0327/0.0318/0.0329/0.0369`。

这已经违反预声明门槛“fold1 与 fold2 任一折的前三项下降不得超过0.003”，所以无论 fold2 后续结果如何，本方案均不可能获得补跑 fold3--5 的资格。用户看到 fold1 严重退化后明确要求停止服务器 Round24；主脚本、fold2 训练及数据加载进程均已终止，GPU 已释放。fold2 在停止前已开始并留下不完整日志/checkpoint，但没有正式 `result.json`，不得将其作为 fold2 结果，也不得生成或宣称双折均值。

严格结论：Round24 只有完整 fold1 证据，不能宣称已完成双折总体评估；但因 fold1 已不可逆地触发预声明单折退化否决条件，Round24 被提前停止且禁止扩展。现有结果和日志保留在 `results/round_024_full_consistency_stochastic_pair/twofold_seed2000/`，不删除、不覆盖。
