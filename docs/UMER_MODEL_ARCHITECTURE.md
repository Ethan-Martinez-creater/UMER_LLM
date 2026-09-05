# UMER 模型组成

> 本文档从源码出发具体描述 UMER 的输入表示、各分支结构、损失函数与训练算法。
> 源码位置以新工作区为准:`project/optimization_rounds/round_013_joint_trifusion/{model.py,run_fold.py}`、`project/src/rumor_detection/models/original_model.py`、`project/optimization_rounds/round_032_sam/sam.py`、`project/optimization_rounds/round_021_cross_view_consistency/regularization.py`。
> 历史内部代号 `round_032_sam` 仅为可追溯性保留;论文与后续研究中方法名统一为 UMER。

## 1. 模型定位

UMER 是单检查点(binary)社交媒体谣言检测器,输入为 PHEME / Ma-Weibo 提供的事件与对话信息(源帖+240小时内按时间保留的回复),输出事件级 `rumor / non-rumor` 预测。它不使用外部语料、网络证据、用户画像、LLM 或事后集成;推理时只是一个普通前向网络。

整体结构:

```text
事件输入
  ├─ 图分支:节点token Set编码器 ⊕ 一层Transformer图编码器 -> 2维图logits
  ├─ 文本分支:DeBERTa-v3-large 对4种标准视图 -> 2维文本logits
  └─ 跨事件检索分支:事件语义统计 -> 从当前训练折记忆库检索31近邻 -> 1维rumor log-odds
线性融合头(5 -> 2) -> 最终二分类logits
```

三路由联合训练(single joint checkpoint),总损失含最终交叉熵 + 图/文本辅助交叉熵 + 跨视图一致性正则。

## 2. 输入表示

### 2.1 事件图

每个事件是一棵以源帖为根的对话树,240小时窗口内按时间保留帖子。预处理产物为每事件一个 `.pt` 文件,包含:

- `node_feat`: `[max_nodes, 395]`,前 384 维为节点语义特征(文本嵌入),后 11 维为附加节点/用户特征;
- `struct_feat`: `[max_nodes, 1024]`,前 1021 维为邻接签名(adjacency signature),最后 3 维为稠密传播摘要;
- `edge_index`、`num_nodes`、标签等。

(维度切分见 `original_model.py` 的 `extract_node_tokens` 与 `StructureFeatureEncoder`:`node_feats[..., -11:]` 取附加 11 维,`struct_feats[..., -3:]` 取 3 个传播统计量。)

PHEME 共 6,425 事件(2,402 谣言 / 4,023 非谣言),105,238 帖;Ma-Weibo 共 4,664 事件(2,313/2,351),1,771,329 帖。两数据集均保留 source-only 事件(PHEME 677 个,10.54%)。

### 2.2 四种标准文本视图

`build_event_views`(`round_009_multiview_deberta/run_fold.py:41`)把源帖与回复序列化为 4 个保持标签的确定性视图,以 ` [SEP] ` 连接,源帖恒在首位:

1. **full**:源帖 + 全部回复(按时间顺序);
2. **reversed**:源帖 + 逆序回复;
3. **even**:源帖 + 偶数位回复 `replies[::2]`;
4. **odd**:源帖 + 奇数位回复 `replies[1::2]`。

- 训练时每个事件随机抽 2 个不同视图(无放回),两视图均接受文本分支监督;
- 验证/测试时使用全部 4 视图并平均其 logits(`JointEventDataset(all_views=True)`);
- tokenizer 为 DeBERTa 的 `spm.model`(sentencepiece),`max_length=256`,截断不填充,动态 pad。

## 3. 图分支(`OriginalRumorDetector`,graph_model)

文件:`project/src/rumor_detection/models/original_model.py`。两个读出并联:

### 3.1 NodeTokenSetEncoder(置换不变 Set 编码)

- 输入 token:14 维 = 11 维非文本节点/用户特征 + 3 个传播特征(度、深度、相对时间,取自 `struct_feat` 末 3 维);
- `token_mlp`:Linear(14→256) → LayerNorm → GELU → Dropout(0.245) → Linear(256→256) → GELU;
- mean-pool 与 max-pool(均带有效节点掩码)拼接为 512 维,`output_proj`:Linear(512→768) → LayerNorm → Dropout(0.21) 输出 768 维。

### 3.2 OriginalGraphBranch(一层 Transformer 图编码器)

- `node_projection`:Linear(395→768) → LayerNorm → GELU → Dropout;
- `StructureFeatureEncoder`:邻接签名(1021→256)与传播摘要(3→256)分别投影后拼接,再投影到 768 维,与节点投影相加;
- 一个可学习结构 CLS token(`s_cls_token`,768 维,init N(0, 0.02²));
- 一层 `nn.TransformerEncoderLayer`(d_model=768, nhead=8, FFN=1536, GELU, pre-norm, dropout=0.35),`src_key_padding_mask` 屏蔽填充节点;
- 读出:CLS 输出 ⊕ 节点均值池化(1536 维)→ Linear → LayerNorm → GELU → Dropout → 768 维事件特征。

### 3.3 融合与分类头(图分支内部)

- Set 读出 768 ⊕ Transformer 读出 768 = 1536 → `fusion_proj`(Linear 1536→768 + LayerNorm + Dropout)→ `classifier`(LayerNorm → Dropout → Linear 768→2)→ **2 维图 logits**;
- `d_model = hidden_dim × 3 = 768`(hidden_dim=256);
- 图分支整体参数量 8,526,850(`train_original.py:78` 硬校验)。

## 4. 文本分支(text_model)

- `AutoModelForSequenceClassification.from_pretrained(deberta-v3-large, num_labels=2)`,本地离线权重,序列分类头输出 **2 维文本 logits**;
- 训练时 forward 接收 `view_count=2` 的成批视图(batch×2 拼接后一次前向),`sample_view_subset` 负责视图采样;
- 文本分支的监督施加在(采样的)每个视图上,不是只在视图均值上(见 §6 损失)。

## 5. 跨事件检索分支(memory)

`JointTriFusionModel.event_semantic_features` + `retrieve`(`round_013/.../model.py:202-232`):

1. **查询向量**(1536 维):取 `node_feat` 前 384 维文本嵌入,计算源帖向量、均值、最大值、均值-源帖差 4 个块(各 384 维,每块先 L2 归一化再拼接,整体再归一化);
2. **记忆库**:仅由当前外层训练折事件构建(`memory_features`/`memory_labels`,buffer 注册,随模型保存);训练查询时排除自身位置(leave-one-out);验证/测试事件从不写入记忆库;
3. **检索**:余弦相似度取 top-k,k=31;softmax 权重温度 0.05;与近邻标签加权得 rumor 概率,clamp 到 [1e-5, 1-1e-5] 后取 logit,输出 **1 维检索 logit**;
4. `retrieve_with_evidence`(`@torch.no_grad`)额外暴露 indices/similarities/weights/labels,供解释性审计使用,与主前向完全隔离。

## 6. 融合与损失

### 6.1 线性融合头

`nn.Linear(5, 2)`:输入 = [图 logits(2), 文本 logits(2), 检索 logit(1)]。clean 初始化为随机;warm_start 时按 late-fusion 对称初始化(复现 0.5g+0.5d+1.0r+0.8 的验证选择起点)。代码中还实现了 `reliability_gate` / `conditional_residual` 两种融合变体,但正式 UMER 用 `linear`。

### 6.2 总损失(`JointTriFusionModel.loss`)

```text
L_total = CE(final_logits, y)                    # 主损失
        + 0.15 · CE(graph_logits, y)             # 图分支辅助
        + 0.15 · CE(每个受监督文本视图的logits, y)  # 文本分支辅助(逐视图,非均值)
        + 0.5  · JS(视图分布)                     # 跨视图一致性
```

- 类别权重:平方根逆频率 `w = (n_0/n_1)^0.5`(PHEME fold1 为 [1.0, 1.2944]),施加于全部 CE;
- **跨视图一致性**(`round_021/.../regularization.py`):广义 Jensen–Shannon 散度——对 [batch, views, 2] 的 logits 取 log_softmax,以各视图概率均值为混合分布,计算 `Σ_v KL(p_v ‖ p̄)` 的平均;施加于训练采样的两个视图;
- LLM 时代补充的可选认知蒸馏头(`cognitive_head`/`node_cognitive_head`)默认 `dim=0`,正式 UMER 不启用,相关钩子不影响前向。

## 7. 优化(exact-replay SAM)

- 优化器:AdamW,weight_decay=0.01;分组学习率:图分支 1e-4、DeBERTa 2e-5、融合头 1e-3;
- 调度:线性 warmup(10% 总步数)+ 线性衰减,horizon=30 epochs;
- 物理 event batch 4 × 梯度累积 16 = **有效 event batch 64**;训练事件每步 2 个文本视图,即每物理步 8 个序列;
- AMP(fp16 autocast + GradScaler);
- max_epochs=30,patience=7(以验证 checkpoint key 连续不升计数)。

### 7.1 Exact-replay SAM(rho=0.05)

SAM 要求两遍使用同一 batch,与梯度累积冲突。Round32 的实现(`sam.py` + `run_fold.py:1033-1083`)按**有效 batch 窗口**执行:

1. 缓存连续 16 个已完成随机图增强与文本视图采样的微批次;
2. 第一遍逐微批次前向/反向,按 `loss/16` 累计;
3. 手动反缩放 AMP 梯度(不推进 GradScaler 状态),以全可训练参数全局梯度范数构造半径 0.05 的扰动并加到参数上;
4. 清空第一遍梯度但保存参数精确原值;
5. **重放**同一 16 个微批次:逐微批次恢复第一遍前的 CPU/CUDA PyTorch RNG 状态,使 dropout 掩码与视图采样完全一致,在扰动参数处累计第二遍梯度;
6. 精确恢复原参数,仅对第二遍梯度做范数 1.0 分组裁剪;
7. AdamW、GradScaler、scheduler 各更新一次;第一遍不触碰任何优化器状态。

效果:训练计算量约 2×,推理零额外成本;训练中的随机性(图增强、视图选择、dropout)在两遍间被精确重放。

## 8. 数据划分与选择

- 外层:五折事件级分层划分,partition seed **3090**(PHEME 与 Ma-Weibo 同种子);
- 内层:外层训练池的 10%,标签分层(`inner_split_strategy=label`),用于 checkpoint 选择与阈值选择;PHEME fold1 为 4626 训练 / 514 验证;
- 训练种子 **2000**(控制初始化、数据顺序、视图采样、图增强、dropout);
- checkpoint 选择规则 `legacy_four_metric_min`:validation-only 排序键 = (min(ACC, W-F1, Macro-F1, Rumor-F1), Macro-F1, ACC) 最大化;
- 阈值:仅在内层验证集上选取(PHEME fold1=0.565,各折在 0.25–0.71 之间);
- held-out 外层测试折只在训练与 checkpoint 选择全部完成后评估一次;
- 交付物为每折一个 checkpoint(`best_joint_model.pt`,含 model_state_dict/threshold/fold/seed/retrieval 参数),无 checkpoint 平均(代码支持 top-k 平均但正式关闭,k=1)、无测试时融合。

## 9. 推理

验证/测试:全部 4 个标准视图、batch=4、无图增强、无 dropout;文本 logits 为 4 视图均值;检索分支查询记忆库(测试事件不在库中);融合后 softmax 取 rumor 概率,与该折验证阈值比较。`retrieve_with_evidence` 可导出近邻证据用于事后解释审计。
