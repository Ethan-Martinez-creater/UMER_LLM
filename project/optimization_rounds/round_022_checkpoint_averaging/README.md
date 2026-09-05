# round_022_checkpoint_averaging（两折筛选完成，不扩展）

Round 021/fold2 暴露出验证最优 checkpoint 到测试集的泛化落差。本目录先准备单轨迹 checkpoint 参数平均的纯函数：沿用当前四指标最小值优先的验证排序，选取同一次训练中验证最强的若干 checkpoint，并只在参数空间合并为一个 checkpoint；推理仍是一个整体模型，不做多模型 logit 集成。

该候选不声称复现 SWA：当前 runner 使用 10% warmup 后线性衰减，而原始 SWA 使用恒定或周期学习率。只有 Round 021 seed2000 完整五折未达标后，才允许接入 runner，并固定候选数、平均规则和选择协议，在 seed2000 的 fold1、fold2 同时筛选；测试集不参与 checkpoint 排名或平均方案选择。

固定筛选入口为 `run_twofold_screen.sh`：seed 2000、fold1/2、`checkpoint_average_top_k=3`，其余训练参数与 Round021 完全一致。每折先按验证四指标排序保留三个同轨迹状态，再分别验证 top2/top3 参数平均；最终只保存验证排序最强的单一 checkpoint。Round021 五折汇总后已按此协议完成两折筛选。

## 两折筛选结果

fold1 在 epoch 13 早停，验证排序最终仍选择 epoch 6 单 checkpoint；测试为 Accuracy 0.8949、Weighted-F1 0.8959、Macro-F1 0.8900、Rumor-F1 0.8667。fold2 在 epoch 15 早停，最终仍选择 epoch 8 单 checkpoint；测试为 0.8926/0.8935/0.8875/0.8634。两折的 top2/top3 参数平均在验证集上都不如最佳单 checkpoint，故本轮没有任何测试结果来自参数平均 checkpoint。

与 Round021 对应两折相比，fold1 四项分别变化 -0.0086/-0.0079/-0.0076/-0.0065，fold2 分别变化 +0.0117/+0.0116/+0.0123/+0.0150；两折均值从 0.8922/0.8929/0.8864/0.8608 变为 0.8938/0.8947/0.8887/0.8650。虽然均值四项略升，但 fold1 全降、fold2 全升，且两个 fold 均拒绝参数平均，因此不能证明 checkpoint 参数平均带来稳定增益。该候选不扩展 fold3--5 或 seed 3090/5090。服务器证据为 `results/round_022_checkpoint_averaging/twofold_seed2000/twofold_comparison.json` 及两折各自的 `result.json`。

## 可复现性限制

Round022 fold1 与 Round021 fold1 使用相同 training seed、partition、数据和训练超参数，但 epoch6 曲线并不相同：Round021 验证为 0.8813/0.8825/0.8759/0.8501，Round022 为 0.8949/0.8960/0.8903/0.8676。核查 `seed_all` 后确认它设置了 Python、NumPy、PyTorch 与 CUDA seed，但没有启用 CUDA 确定性算法。因此本轮测试变化同时包含同 seed 的非确定性重跑波动和 checkpoint 参数平均，不能把全部差值归因于参数平均。两折筛选与后续三 training seed 完整实验仍按实际测试结果判断，但报告必须保留该限制。

原门控要求 fold1/2 测试均相对 Round021 有稳定增益；本轮未满足该门控，因此 `run_seed2000_remaining_folds.sh` 与 `run_additional_seed_fivefold.sh` 均未启动。
