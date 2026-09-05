# round_009_multiview_deberta

方案标识：`round_009_multiview_deberta`

针对 round 008 固定截断前 256 tokens 和明显过拟合的问题，本轮为每个事件构造四个确定性视图：source + 正序 replies、source + 逆序 replies、source + 偶数 replies、source + 奇数 replies。训练时每次随机选择一个视图；验证和测试时平均四个视图的 logits。

模型使用等效 batch 64、DeBERTa 学习率 `2e-5`、最多 60 epoch、patience 7。checkpoint 与阈值均只在验证集选择，选择目标为 Accuracy、Weighted-F1、Macro-F1、Rumor-F1 四项中的最小值，直接对应最终门槛；测试集仅最终评估一次。

fold 1 / seed 2000 结果为 Accuracy `0.8685`、Weighted-F1 `0.8696`、Macro-F1 `0.8621`、Rumor-F1 `0.8325`。相对 round 008 同折分别提升约 `0.0031`、`0.0037`、`0.0047`、`0.0089`，证明多窗口有小幅帮助，但仍明显低于门槛；本轮不进入完整五折。
