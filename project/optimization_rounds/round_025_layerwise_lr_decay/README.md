# round_025_layerwise_lr_decay

方案标识：`round_025_layerwise_lr_decay`

## 立项依据

Round21 是当前 PHEME 最强的单一联合模型，五折为 `0.8951/0.8955/0.8890/0.8629`，但 fold1/fold2 的验证到测试差距明显，尤其 fold2 验证 Accuracy 约0.9086而测试仅0.8809。Round23/24 已证明继续扩大四视图分类或一致性覆盖不能解决该问题。

当前 DeBERTa-v3-large 的 embedding、24个 encoder layer、pooler 与 classifier 全部使用同一个 `2e-5` 学习率。在约72%的 PHEME 外层训练数据上，这会让底层通用语言表示与顶层任务表示以相同幅度更新。本轮只加入层级学习率衰减：classifier/pooler 保持 `2e-5`；从顶层 encoder 向底层和 embedding 按固定系数 `0.95` 逐层衰减。图分支 `1e-4`、融合层 `1e-3`、weight decay、scheduler、warmup、类别权重和所有数据协议保持 Round21 不变。

本方案不冻结编码器、不更换预训练模型、不增加损失、不组合 checkpoint，最终仍为 clean 初始化后联合训练得到的单一 checkpoint。方案由 Round21 的小样本验证—测试落差和当前优化器参数分组直接推导；AnySearch 在2026-07-22连接失败，因此立项阶段未读取或引用新论文，后续若检索成功必须先登记文献台账。

## 强制 fold1+fold2 双折筛选

固定 `partition_seed=3090`、`training_seed=2000`，连续完成 fold1、fold2，配置为 Round21 的 `train_view_count=2`、`classification_view_count=0`（即两个编码视图全部参与分类）、`view_consistency_weight=0.5`、物理 event batch=4、梯度累积16、等效 event batch64、最多30 epoch、patience7；唯一变化为 `text_layerwise_lr_decay=0.95`。

预声明门槛与 Round24 一致：

1. fold1 与 fold2 的 Accuracy、Weighted-F1、Macro-F1 任一项相对 Round21 同折下降不得超过 `0.003`；
2. 二折 Accuracy、Weighted-F1、Macro-F1 均值必须全部严格高于 Round21 二折均值；
3. 二折 Macro-F1 均值至少提高 `0.005`；
4. 两折必须全部完成后才能判断方案；即使 fold1 单折较高也不得提前扩展；
5. 双折脚本结束后只生成对比报告，不自动运行 fold3--5。

Round21 二折基线均值为 Accuracy `0.8922`、Weighted-F1 `0.8929`、Macro-F1 `0.8864`、Rumor-F1 `0.8608`。

计划输出目录：`results/round_025_layerwise_lr_decay/twofold_seed2000/`。

## 启动记录

2026-07-22 已完成服务器 Shell/Python 语法、CLI 参数、24层配置与8项单元测试核验。随后启动 fold1→fold2 连续任务，后台主 PID `1917556`；启动参数确认 `train_view_count=2`、`classification_view_count=0`、`text_layerwise_lr_decay=0.95`。fold1 初始训练显存约14.2 GiB，无即时 OOM。两折结束后脚本只生成双折报告，不自动运行fold3--5。

## 正式双折结果与结论

Round25 已完整完成 fold1、fold2 并自动停止：

| fold | Accuracy | Weighted-F1 | Macro-F1 | Rumor-F1 | 最佳epoch | 阈值 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.8934 | 0.8933 | 0.8861 | 0.8571 | 5 | 0.705 |
| 2 | 0.8848 | 0.8857 | 0.8791 | 0.8529 | 6 | 0.705 |
| 二折均值 | 0.8891 | 0.8895 | 0.8826 | 0.8550 | — | — |

相对 Round21 同折，fold1 四项下降 `0.0101/0.0105/0.0116/0.0161`，fold2 四项提升 `0.0039/0.0038/0.0040/0.0045`。二折均值则下降 `0.0031/0.0033/0.0038/0.0058`。

预声明门槛全部未通过：fold1 前三项下降超过0.003；二折 Accuracy、Weighted-F1、Macro-F1 未同步提升；Macro-F1 二折均值没有达到+0.005，反而下降0.0038。因此 `eligible_to_run_folds_3_to_5=false`，禁止补跑剩余三折。

严格结论：0.95层级学习率衰减改善了原先较弱的fold2，却损害原先较强的fold1，不能稳定替换统一2e-5学习率的Round21。后续不以更强衰减或冻结更多底层的方式重复该路线。权威报告：`results/round_025_layerwise_lr_decay/twofold_seed2000/twofold_comparison.json`。
