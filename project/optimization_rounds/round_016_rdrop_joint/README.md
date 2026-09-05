# round_016_rdrop_joint

方案标识：`round_016_rdrop_joint`

本轮保持 Round 014 的任务级 clean 初始化、标准随机单视图训练/四视图推理、严格训练折 leave-one-out 检索记忆、单 checkpoint 和三分支联合训练协议。每个训练批次对同一已选文本视图进行两次带 dropout 的完整前向传播，监督损失取均值，并在两个最终二分类分布之间加入对称 KL；评估仍为一次确定性前向，不构成模型集成。

依据为 [R-Drop: Regularized Dropout for Neural Networks（NeurIPS 2021）](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)。论文用双向 KL 约束同一输入经不同 dropout 子模型得到的输出一致性。本项目只采用这一正则机制，不沿用论文的具体任务设置；首轮固定 `rdrop_weight=1.0`，避免在单个测试折上搜索超参数。

筛选协议仍为 seed 2000 / fold 1：验证集选择 checkpoint 和阈值，测试集只在训练结束后评估一次。只有相对 Round 014 的四项指标出现充分提升，才考虑扩展完整五折。

## 正式筛选结果（2026-07-17）

训练在 epoch 7 取得验证集最优：Accuracy 0.9047、Weighted-F1 0.9051、Macro-F1 0.8991、Rumor-F1 0.8753，阈值为 0.40；epoch 14 连续七轮未改进后早停。恢复 epoch 7 的单一 checkpoint 后，测试结果为 Accuracy 0.8825、Weighted-F1 0.8826、Macro-F1 0.8748、Rumor-F1 0.8438。

相对 Round 014，测试 Accuracy/Weighted-F1 分别提高约 0.0039/0.0027，但 Macro-F1 仅提高约 0.0014，Rumor-F1 下降约 0.0041。R-Drop 缓解了整体错误率，却未解决少数类识别和验证—测试差异，因此本轮否决，不扩展完整五折。服务器证据：`results/round_016_rdrop_joint/screen/fold_1_train_2000/result.json`。

随后使用 `analysis/analyze_joint_threshold_reachability.py` 对固定 checkpoint 做仅诊断的测试阈值 oracle 审计。即使允许在测试集搜索阈值，最佳四指标最小值也只有 0.8485；阈值 0.145 时四项为 0.8786/0.8799/0.8736/0.8485，无法全部严格超过 0.88。验证所选阈值 0.40 的混淆矩阵为 TN=726、FP=78、FN=73、TP=408。该 oracle 不用于选模或后续超参数，结论仅是当前 checkpoint 的瓶颈不可能由阈值校准解决。服务器证据：`analysis/joint_threshold_reachability/round_016_seed2000_fold1.json`。
