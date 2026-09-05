# round_013_joint_trifusion

方案标识：`round_013_joint_trifusion`

本轮回应后期融合不可作为单一模型交付的问题。它复用严格同折训练好的原图模型与 Round 009 DeBERTa-v3-base 多视图模型作为初始化，但随后在一个 `JointTriFusionModel` 中联合微调：最终分类损失同时反向传播到图分支、DeBERTa 分支和融合层，并以较小的图/文本辅助分类损失防止分支退化。

训练折 MiniLM 事件语义表示及标签作为 buffer 保存进同一个 checkpoint。模型前向时从节点特征构造 source/mean/max/mean-source 查询，执行 `k=31, temperature=0.05` 的语义记忆检索；训练事件屏蔽自身，验证和测试只能查询训练折。最终 checkpoint 因此包含两条神经分支、可训练融合层和严格训练折检索记忆，测试时只加载这一份 checkpoint 完成一次整体推理。

训练物理 batch 8、累计 8，等效 batch 64；图与 DeBERTa 学习率 `5e-6`，融合层 `5e-4`。验证集选择四项指标最小值最高的 checkpoint 和阈值，测试集只用于最终评估。

与已有方法的实质差异：Round 007 是 MiniLM 与 DeBERTa 的中间特征层次融合且没有图分支/训练折检索；此前三路结果是冻结 checkpoint 后的手工 logit 组合。本轮是单一模块、联合梯度和单 checkpoint，不把后期集成指标冒充整体模型指标。

## seed 2000 / fold 1 结果

完整训练在 epoch 1 取得最佳验证结果，随后连续 5 epoch 未改善并早停。最终只加载 `best_joint_model.pt` 一次完成测试推理：Accuracy `0.8918`、Weighted-F1 `0.8920`、Macro-F1 `0.8849`、Rumor-F1 `0.8566`。相较冻结三路后期融合的 `0.8887/0.8892/0.8824/0.8551` 四项均小幅提高，证明联合训练有效；但 Rumor-F1 距离 `>0.88` 仍约 0.0234，本轮不扩展至完整五折三 seed。
