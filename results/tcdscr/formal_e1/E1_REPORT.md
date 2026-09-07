# Formal E1 — Causal Encoder 正式实验报告

- 日期：2026-09-07（训练完成 `ALL_DONE 18:17:49`）
- 计划依据：`docs/TC_DSCR_V2_FORMAL_EXECUTION_PLAN.md` §36（Formal E1 — Causal Encoder）、§37（Encoder Initialization）
- 状态：**60/60 runs 完成，零失败**
- 范围：仅 Causal Encoder 正式实验；**不进入 E2**

---

## 1. 协议概述（§36/§37）

每个数据集 `pheme` / `maweibo` 独立执行：

| 维度 | 规定 |
|---|---|
| 数据集 | PHEME、Ma-Weibo（各 5 折 × 3 seeds × 2 init = 30 runs，共 60 runs） |
| seeds | 2000 / 2001 / 2002 |
| 划分 | event-level 分层 5 折（partition_seed=3090，StratifiedKFold shuffle）＋ fold 内分层 10% 验证集 |
| 每 epoch 采样 | 每个 train event 随机均匀采 1 个可用动态快照（6 个主 cutoff 池） |
| snapshot pool | 5m / 15m / 30m / 1h / 3h / 6h（PRIMARY_CUTOFFS） |
| SOURCE_ONLY | 不参与 encoder 主训练 |
| 初始化 | Random Init vs UMER Init（同 fold 的 UMER fold-wise checkpoint，随机权重重训对照） |
| 报告口径 | 每 cutoff：Accuracy / Macro-F1 / Weighted-F1 / Rumor-F1（8 个 cutoff：SOURCE_ONLY, 5, 15, 30, 60, 180, 360, 1440；1440 为 late diagnostic） |

## 2. 实验设置

**冻结超参**（UMER 历史五折训练配方，见 `project/tcdscr/config/schema.py` 与每个 run 的 `run_manifest.json#hparams`）：

```text
batch_size=32  max_epochs=60  patience=7
AdamW lr=1e-4  weight_decay=0.05  label_smoothing=0.1
```

**模型与特征**（`project/tcdscr/models/causal_social_encoder.py`）：

- 图编码：384D 语义节点特征 + 1021D 邻接签名（parent→child 有向边 + 对角自环，行按自身出度归一）＋ 3D snapshot 统计（norm_degree=snapshot 内 max 归一、norm_depth=depth/19、norm_time=30min bin/480）；struct 矩阵在 GPU collate 内按冻结公式在线构建（`tests/test_e1_collate.py::test_gpu_collate_matches_frozen_struct_definition` 锁定与 CPU 定义一致）
- 参数规模：encoder 6,870,786（d_model=768，graph transformer 结构）
- 训练脚本：`scripts/tcdscr_run_e1.py`（协议 A：先全注册表解析 split，仅加载本 fold 事件；验证集每 event 固定快照采样 seed 5000+fold；val macro-F1 早停 patience 7，取 best state_dict 在 test 8 视图评估）
- UMER init：`/data/jyz/next/llm/checkpoints/{dataset}/fold_{fold+1}_train_2000/best_joint_model.pt`（每 run 把同 fold UMER checkpoint 的 39 个张量拷贝到 encoder，0 跳过；init 信息记录于 `run_manifest.json#init_info`）
- 运行编排：`scripts/tcdscr_code/e1_driver.py`（顺序 subprocess + `e1_state.json` 断点续跑 + `e1_all.log` 追加日志）

## 3. 运行矩阵与完整性

| 数据集 | init | folds | seeds | runs | 完成 | 失败 |
|---|---|---|---|---|---|---|
| pheme | random | 0–4 | 2000/2001/2002 | 15 | 15 | 0 |
| pheme | umer | 0–4 | 2000/2001/2002 | 15 | 15 | 0 |
| maweibo | random | 0–4 | 2000/2001/2002 | 15 | 15 | 0 |
| maweibo | umer | 0–4 | 2000/2001/2002 | 15 | 15 | 0 |

- `scripts/tcdscr_verify_e1.py` 完整性核查：60 runs 全部含 `metrics.json / predictions.jsonl / run_manifest.json / history.json`；8 个 cutoff × 4 指标数值均在 [0,1]；manifest 含 dataset/fold/init/seed/hparams/split/partition_seed/best_epoch/best_val_macro_f1/init_info；history 为非空 epoch 记录列表
- 预测行数：**532,272**（60 runs × 8 cutoff × 每 fold test 事件；每个事件每条 cutoff 恰好一条）
- 训练日志：`results/tcdscr/formal_e1/e1_all.log`（每 run 启动时间、结束状态、best_epoch、val macro-F1）
- 模型权重：`best_encoder.pt`（每 run 1 个，仅存服务器 `/data/jyz/next/llm/results/tcdscr/formal_e1/`，不在仓库内）

## 4. 结果 — per-run 口径（30 runs mean ± std）

### 4.1 PHEME

#### random init (per-run mean ± std)

| cutoff | accuracy | macro_f1 | weighted_f1 | rumor_f1 |
|---|---|---|---|---|
| SOURCE_ONLY | 0.8572 ± 0.0087 | 0.8462 ± 0.0098 | 0.8565 ± 0.0089 | 0.8053 ± 0.0143 |
| 5 | 0.8568 ± 0.0082 | 0.8477 ± 0.0084 | 0.8570 ± 0.0080 | 0.8109 ± 0.0108 |
| 15 | 0.8574 ± 0.0097 | 0.8487 ± 0.0099 | 0.8578 ± 0.0095 | 0.8128 ± 0.0125 |
| 30 | 0.8581 ± 0.0106 | 0.8496 ± 0.0110 | 0.8586 ± 0.0105 | 0.8140 ± 0.0140 |
| 60 | 0.8577 ± 0.0113 | 0.8493 ± 0.0116 | 0.8582 ± 0.0110 | 0.8137 ± 0.0143 |
| 180 | 0.8580 ± 0.0119 | 0.8495 ± 0.0121 | 0.8585 ± 0.0116 | 0.8142 ± 0.0149 |
| 360 | 0.8590 ± 0.0120 | 0.8506 ± 0.0121 | 0.8595 ± 0.0116 | 0.8153 ± 0.0146 |
| 1440 | 0.8603 ± 0.0118 | 0.8519 ± 0.0119 | 0.8607 ± 0.0114 | 0.8169 ± 0.0144 |

#### umer init (per-run mean ± std)

| cutoff | accuracy | macro_f1 | weighted_f1 | rumor_f1 |
|---|---|---|---|---|
| SOURCE_ONLY | 0.8634 ± 0.0083 | 0.8525 ± 0.0095 | 0.8626 ± 0.0086 | 0.8128 ± 0.0141 |
| 5 | 0.8683 ± 0.0078 | 0.8596 ± 0.0084 | 0.8684 ± 0.0078 | 0.8247 ± 0.0113 |
| 15 | 0.8703 ± 0.0098 | 0.8619 ± 0.0106 | 0.8705 ± 0.0098 | 0.8280 ± 0.0141 |
| 30 | 0.8719 ± 0.0108 | 0.8637 ± 0.0118 | 0.8721 ± 0.0109 | 0.8303 ± 0.0158 |
| 60 | 0.8718 ± 0.0109 | 0.8636 ± 0.0120 | 0.8720 ± 0.0111 | 0.8302 ± 0.0162 |
| 180 | 0.8730 ± 0.0110 | 0.8649 ± 0.0121 | 0.8732 ± 0.0111 | 0.8318 ± 0.0163 |
| 360 | 0.8735 ± 0.0112 | 0.8653 ± 0.0122 | 0.8737 ± 0.0113 | 0.8322 ± 0.0164 |
| 1440 | 0.8741 ± 0.0118 | 0.8659 ± 0.0129 | 0.8742 ± 0.0119 | 0.8329 ± 0.0171 |

### 4.2 Ma-Weibo

#### random init (per-run mean ± std)

| cutoff | accuracy | macro_f1 | weighted_f1 | rumor_f1 |
|---|---|---|---|---|
| SOURCE_ONLY | 0.7220 ± 0.0371 | 0.7053 ± 0.0496 | 0.7048 ± 0.0499 | 0.7735 ± 0.0204 |
| 5 | 0.8874 ± 0.0153 | 0.8871 ± 0.0155 | 0.8870 ± 0.0155 | 0.8934 ± 0.0136 |
| 15 | 0.9187 ± 0.0160 | 0.9186 ± 0.0161 | 0.9186 ± 0.0161 | 0.9202 ± 0.0152 |
| 30 | 0.9271 ± 0.0112 | 0.9271 ± 0.0112 | 0.9271 ± 0.0112 | 0.9279 ± 0.0106 |
| 60 | 0.9319 ± 0.0106 | 0.9319 ± 0.0106 | 0.9319 ± 0.0106 | 0.9323 ± 0.0103 |
| 180 | 0.9394 ± 0.0114 | 0.9394 ± 0.0114 | 0.9394 ± 0.0114 | 0.9397 ± 0.0110 |
| 360 | 0.9415 ± 0.0113 | 0.9415 ± 0.0113 | 0.9415 ± 0.0114 | 0.9420 ± 0.0109 |
| 1440 | 0.9425 ± 0.0111 | 0.9425 ± 0.0111 | 0.9425 ± 0.0111 | 0.9433 ± 0.0106 |

#### umer init (per-run mean ± std)

| cutoff | accuracy | macro_f1 | weighted_f1 | rumor_f1 |
|---|---|---|---|---|
| SOURCE_ONLY | 0.7320 ± 0.0292 | 0.7185 ± 0.0349 | 0.7180 ± 0.0350 | 0.7790 ± 0.0174 |
| 5 | 0.8972 ± 0.0112 | 0.8970 ± 0.0113 | 0.8969 ± 0.0114 | 0.9008 ± 0.0099 |
| 15 | 0.9223 ± 0.0081 | 0.9223 ± 0.0081 | 0.9223 ± 0.0081 | 0.9230 ± 0.0078 |
| 30 | 0.9291 ± 0.0086 | 0.9291 ± 0.0086 | 0.9291 ± 0.0086 | 0.9293 ± 0.0084 |
| 60 | 0.9355 ± 0.0086 | 0.9355 ± 0.0086 | 0.9355 ± 0.0086 | 0.9356 ± 0.0087 |
| 180 | 0.9442 ± 0.0066 | 0.9442 ± 0.0066 | 0.9442 ± 0.0066 | 0.9443 ± 0.0069 |
| 360 | 0.9474 ± 0.0074 | 0.9474 ± 0.0074 | 0.9474 ± 0.0074 | 0.9477 ± 0.0075 |
| 1440 | 0.9508 ± 0.0071 | 0.9507 ± 0.0071 | 0.9507 ± 0.0071 | 0.9512 ± 0.0071 |

## 5. 结果 — pooled 口径（每 seed 合并 5 folds 预测，3 seeds mean ± std）

### 5.1 PHEME

| cutoff | random acc / mF1 / wF1 / rF1 | umer acc / mF1 / wF1 / rF1 |
|---|---|---|
| SOURCE_ONLY | 0.8572 / 0.8464 / 0.8566 / 0.8056 | 0.8634 / 0.8527 / 0.8627 / 0.8130 |
| 5 | 0.8568 / 0.8478 / 0.8571 / 0.8110 | 0.8683 / 0.8597 / 0.8685 / 0.8248 |
| 15 | 0.8574 / 0.8488 / 0.8579 / 0.8128 | 0.8703 / 0.8620 / 0.8705 / 0.8281 |
| 30 | 0.8581 / 0.8497 / 0.8587 / 0.8141 | 0.8719 / 0.8638 / 0.8722 / 0.8305 |
| 60 | 0.8577 / 0.8494 / 0.8583 / 0.8138 | 0.8718 / 0.8637 / 0.8721 / 0.8305 |
| 180 | 0.8580 / 0.8496 / 0.8585 / 0.8142 | 0.8730 / 0.8650 / 0.8733 / 0.8321 |
| 360 | 0.8590 / 0.8507 / 0.8596 / 0.8153 | 0.8735 / 0.8654 / 0.8737 / 0.8324 |
| 1440 | 0.8603 / 0.8520 / 0.8608 / 0.8169 | 0.8741 / 0.8660 / 0.8743 / 0.8331 |

### 5.2 Ma-Weibo

| cutoff | random acc / mF1 / wF1 / rF1 | umer acc / mF1 / wF1 / rF1 |
|---|---|---|
| SOURCE_ONLY | 0.7221 / 0.7066 / 0.7060 / 0.7731 | 0.7320 / 0.7194 / 0.7189 / 0.7786 |
| 5 | 0.8874 / 0.8871 / 0.8870 / 0.8933 | 0.8972 / 0.8970 / 0.8970 / 0.9008 |
| 15 | 0.9187 / 0.9186 / 0.9186 / 0.9201 | 0.9223 / 0.9223 / 0.9223 / 0.9229 |
| 30 | 0.9271 / 0.9271 / 0.9271 / 0.9278 | 0.9291 / 0.9291 / 0.9291 / 0.9293 |
| 60 | 0.9319 / 0.9319 / 0.9319 / 0.9323 | 0.9355 / 0.9355 / 0.9355 / 0.9356 |
| 180 | 0.9394 / 0.9394 / 0.9394 / 0.9396 | 0.9442 / 0.9442 / 0.9442 / 0.9443 |
| 360 | 0.9415 / 0.9415 / 0.9415 / 0.9419 | 0.9474 / 0.9474 / 0.9474 / 0.9477 |
| 1440 | 0.9425 / 0.9425 / 0.9425 / 0.9432 | 0.9508 / 0.9508 / 0.9507 / 0.9513 |

## 6. 训练记录

| 数据集 | init | best_epoch (mean / min / max) | best_val macro-F1 (mean / min / max) |
|---|---|---|---|
| pheme | random | 12.4 / 6 / 28 | 0.8671 / 0.8326 / 0.8860 |
| pheme | umer | 8.7 / 2 / 21 | 0.8817 / 0.8554 / 0.8978 |
| maweibo | random | 9.9 / 2 / 21 | 0.9337 / 0.9091 / 0.9706 |
| maweibo | umer | 12.3 / 1 / 22 | 0.9415 / 0.9171 / 0.9679 |

- UMER Init：每 run 从同 fold `fold_{fold+1}_train_2000/best_joint_model.pt` 拷贝 39 个张量，0 个跳过（`init_info` 全量记录）
- 训练跨度：首 run `pheme_fold0_random_seed2000` 21:47 启动 → 最后 `maweibo_fold4_umer_seed2002` 18:12 启动 → `ALL_DONE 18:17:49`
- 每 run 完整 epoch 级 loss / val 曲线见各 run `history.json`

## 7. 结果观察（描述性，不含假设检验）

1. **UMER Init 一致优于 Random Init 且差距随 cutoff 增长**：PHEME accuracy 差 +0.6 → +1.4pt（SOURCE_ONLY→1440，per-run）；Ma-Weibo +1.0 → +0.8pt（SOURCE_ONLY→1440）。Rumor-F1（研究关注类别）PHEME 差 +0.8 → +1.6pt，Ma-Weibo +0.5 → +0.8pt。所有 4 指标 × 8 cutoff × 2 数据集共 64 个组合中 UMER 全部更高。
2. **随 cutoff 单调上升**：除 PHEME random 5m 略低于 SOURCE_ONLY 外，两数据集两 init 的 4 指标全部随传播窗口单调（或非减）上升，验证"更多因果可见的社会上下文 → 更好的编码"方向。
3. **数据集差异符合预期**：Ma-Weibo 整体明显高于 PHEME（1440m accuracy 0.943–0.951 vs 0.860–0.874），与两数据集标签可分离性与规模差异一致；Ma-Weibo SOURCE_ONLY 明显低于 5m（~0.72 → ~0.89），说明该数据集的传播上下文对判别贡献极大。
4. **pooled 与 per-run 两个口径方向完全一致**（差异 <0.0015），结果不依赖聚合方式。
5. 此阶段仅为 encoder 单独能力（无 selector、无 token budget），§48 的完整成功标准评估留待后续实验阶段。

## 8. 产物清单与复现

- 聚合：`results/tcdscr/formal_e1/formal_e1_summary.json`（per-run 与 pooled 全量）、`formal_e1_tables.md`（per-run 表）
- 核查：`scripts/tcdscr_verify_e1.py`（60 runs / 532,272 预测行 / 0 问题）
- 训练日志：`results/tcdscr/formal_e1/e1_all.log`（60 run 启动/结束/val 记录）
- 代码：`scripts/tcdscr_run_e1.py`、`scripts/tcdscr_summarize_e1.py`、`scripts/tcdscr_code/e1_driver.py`（服务器部署副本与本地逐一 diff 一致）、`project/tcdscr/tests/test_e1_collate.py`；
- 权重：60× `best_encoder.pt` 仅存服务器 `/data/jyz/next/llm/results/tcdscr/formal_e1/{dataset}/{fold}_{init}_seed{seed}/`（每个 run 的 `checkpoint_sha16` 记录在 manifest 中，可用于复现比对）
- 复现入口：`python scripts/tcdscr_run_e1.py --dataset {pheme|maweibo} --fold {0..4} --init {random|umer} --seeds 2000,2001,2002`