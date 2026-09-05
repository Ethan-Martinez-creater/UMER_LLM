# round_006_node_deberta

方案标识：`round_006_node_deberta`

本轮用折内微调的 `microsoft/deberta-v3-base` 替换每个传播节点原有的 384 维 MiniLM 文本向量。DeBERTa 不读取测试标签；每折只使用该折训练集训练，并由验证集选择 checkpoint。随后分别编码 source 与所有保留 replies，按 `graph_final/*.pt` 内保存的 `node_ids` 精确回填节点特征。

保留内容：

- 原始 11 维节点辅助特征及末尾聚合用户分数；
- 原始邻接、时间、深度和度特征；
- 原始图 Transformer、Set 分支、分类头；
- 图训练 batch 64、AdamW `3e-4`、weight decay `3e-4`、最多 60 epoch、patience 7；
- 严格外层五折和固定内层验证集。

DeBERTa 微调使用物理 batch 8、梯度累积 8（等效 batch 64）和 `2e-5` 学习率。中间编码器只按验证集 Macro-F1 选择；最终图模型仍按验证损失 checkpoint 汇报。单折筛选入口为 `screen_fold.py`；只有筛选明显优于此前候选后才接入完整 5 折 × 3 seed 调度。

首个单折诊断曾按文本验证损失选择 encoder，固定图 checkpoint 得到 Accuracy `0.8638`、Weighted-F1 `0.8646`、Macro-F1 `0.8565`、Rumor-F1 `0.8241`。审计发现被选文本 epoch 的验证 Macro-F1 仅 `0.8453`，而后续 epoch 达到 `0.8973`，因此改为验证 Macro-F1 选择中间 encoder 后重新筛选；测试集不参与该选择。

修正版固定图 checkpoint 为 Accuracy `0.8599`、Weighted-F1 `0.8597`、Macro-F1 `0.8501`、Rumor-F1 `0.8117`，仍低于首版和纯文本探针。结论：直接丢弃 MiniLM 并用逐节点 DeBERTa 向量替换，会破坏原图分支已有的稳定语义空间；本轮不进入完整 5 折 × 3 seed。
