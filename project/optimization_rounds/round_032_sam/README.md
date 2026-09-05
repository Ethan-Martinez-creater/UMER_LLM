# Round 032：有效batch级 Sharpness-Aware Minimization

状态：实现与协议测试中，尚无正式结果。这是当前数据范围内最后一次模型优化尝试；双折失败后结束增量优化。

## 唯一变量

保持Round21全部模型与数据配置，只把普通AdamW更新改为SAM包裹的AdamW：

- PHEME；
- `partition_seed=3090`；
- `training_seed=2000`；
- clean单模型联合训练；
- DeBERTa-v3-large；
- 图、文本、训练折语义检索三路线性融合；
- 随机两个标准文本视图；
- 广义JSD权重0.5；
- 平方根逆频率类别权重；
- 物理event batch 4；
- 等效event batch 64；
- 原学习率、AdamW weight decay 0.01和线性warmup/衰减；
- `legacy_four_metric_min` validation-only checkpoint及validation-only阈值；
- `max_epochs=30`、`patience=7`。

SAM扰动半径只使用一个预声明值`rho=0.05`，不做validation/test网格搜索。参考论文与实现来源已记录在`analysis/literature_reference_ledger.md`。

## 等效batch协议

SAM要求第一遍与第二遍使用同一个batch。Round21通过16个物理batch累积成等效batch 64，因此Round32不能对每个物理batch单独扰动。

每次正式更新执行：

1. 缓存连续16个已完成随机图增强和文本视图采样的CPU微批次；
2. 第一遍逐个前向/反向，并按Round21的`loss/16`累计等效batch梯度；
3. 手动还原AMP scale，使用全部可训练参数的全局梯度范数构造半径0.05的SAM扰动；
4. 清空第一遍梯度，但保存参数的精确原值；
5. 重放相同16个微批次，并逐微批次恢复第一遍前向前的CPU/CUDA PyTorch RNG，因此Dropout掩码相同；
6. 在扰动参数处累计第二遍梯度；
7. 精确恢复原参数，只对第二遍梯度执行Round21相同的分组裁剪；
8. AdamW、GradScaler和scheduler各更新一次。

最后不足16个微批次的窗口仍沿用Round21固定除以16的行为，避免改变历史训练语义。第一遍SAM扰动不调用AdamW，不初始化或推进AdamW状态。

## 资源

- 两遍前后向顺序执行，激活显存不翻倍；
- 为精确恢复参数，需要临时保存一份可训练参数原值，预计增加约一个FP32模型参数副本的显存；
- 训练计算量接近Round21的两倍；
- 推理仍是一个普通单checkpoint，不增加推理成本。

## 双折门槛

只连续运行fold1+fold2，脚本不会自动运行fold3--5。相对Round21同折：

- 任一折任一门槛指标下降超过0.003：否决；
- 双折Accuracy、Weighted-F1、Macro-F1均值必须全部提高；
- 双折Macro-F1均值提升至少0.005。

通过后仍需用户同意才补齐fold3--5。失败则按用户要求结束当前模型优化并进入下一项工作。

