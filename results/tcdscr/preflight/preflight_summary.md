# TC-DSCR Preflight Summary

**执行日期**:2026-09-06 · **执行方式**:本地审计 + 服务器(DGPA /data/jyz/next/llm/results/tcdscr/preflight/)只读数据审计与重建实验
**代码版本**:git commit `572d5506880eb361af907a855f6736509c8a8296`
**约束遵守**:未训练模型、未改 UMER 结构、未换数据集/embedding、未修改旧 `.pt`、未自行放宽任何门槛。

## Overall Status

**PARTIAL** — PHEME 侧六问全部可回答且 parity 差异完全可解释、原型无泄漏;Weibo22(KPG v1 release)在时间维度上不可用;存在 4 项 blocking issues 与 5 项需研究审批的决策点。按协议不判 PASS。

## PHEME

- **temporal readiness: PASS** — 6,425 事件、104,582 节点全量审计:source timestamp coverage **100%**、reply timestamp coverage **100%**、chronology violation **0**;事件时长 P50≈2.6h / P90≈35h / P99≈192h;节点数 P50/P90/P99 = 12/31/99。
- **parent readiness: FAIL(门槛)** — parent relation coverage **98.97%**(1,017 条回复无 `in_reply_to_status_id_str`,占回复 1.56%);orphan(parent 指向事件外)1.76%。现有 UMER 管线将这些节点按无边孤立点处理,因此不阻碍重建,但未达预注册的 ≥99.9% 门槛。
- **snapshot readiness: PASS** — 传播高度前置:1h 时平均节点 10.46(全窗 16.4),reply 累计覆盖 P50:1h=80%、3h=95.8%、6h=100%;24h 后平均节点 15.52。可支撑 1h/6h/24h 等多快照,但早窗口与全窗差异主要发生在 0–6h。

## Weibo22

- **download source**: github.com/kkkkk001/KPG,release tag **v1**(commit `8b1d16b3c0…`),main HEAD `7b41f6647f…`;经本地代理下载 raw 文件(服务器无 GitHub 访问),SHA256:`Weibo_label_All.zip` `451e51bc…`、`data.TD_RvNN.vol_5000.zip` `f58ab42e…`;原始 zip 保留于本地 `data_raw/weibo22/`(不入库)。
- **temporal readiness: FAIL** — 两个文件均为经典 TD_RvNN/BiGCN 格式:label 文件仅 `label<TAB>topic<TAB>weibo_id`,树文件为 `weibo_id<TAB>父index链<TAB>词index:词频`。**任何位置都不存在 absolute timestamp**(source 或节点)。
- **propagation readiness: 部分** — 4,174 棵根树(标签 true/false 各 2,087,完全平衡;topic 仅 covid-19/other 两类)、961,962 节点行、parent-child 拓扑完整、无坏行;但文本只有 vol_5000 词表 index,**无原文**。
- **global chronology readiness: FAIL** — 无 source absolute time,无法做任何 chronological split。
- **判决:`NOT_READY_WEIBO22_TEMPORAL`**(按协议不自行换数据集,见待审批问题 1)。

## UMER Feature Pipeline

- **384D**: `paraphrase-multilingual-MiniLM-L12-v2` 编码 `clean_tweet_pheme` 后文本;`embeddings.py` + `text_cleaning.py` + `pipeline.py` steps 2/4。
- **11D**: `node_feat[:,384:395]` = 10 维用户特征(reposts/comments(恒0)/attitudes/followers/statuses/favourites/bi_followers(恒0)/verified/geo/created_at_norm,adapter `_extract_record` L104-136 + `user_features.py` L13-163)+ 1 维情感(`sentiment.py`,**存储值为 (raw+1)/2 归一化**)。`.pt` 实际 396 列(尾列 userFeatures 聚合分,模型切片 `[:, :395]` 不使用)。
- **1021D**: 邻接签名 = 节点按 elapsed_seconds 排序,`adj[parent,child]=1` + 对角自环,行除以自身出度,pad/截断至 1021(`preprocess_graph.py` L57-76);边构建 child→parent(`graph_builder.py` L111-125)。
- **3D**: `[norm_degree, norm_depth, norm_timestep]` — norm_depth=depth/19(固定分母,BFS clamp 19);norm_timestep=time_bin/480(**30 分钟 bin,参考 `.pt` 的 `max_time_steps=480` 实证**);norm_degree=(出度-1)/事件内 max(**依赖全事件节点,见待审批问题 2**)。
- **240h cutoff 实现位置**: `time_windows.assign_time_bins`(elapsed∈[0,240h] **过滤**而非 clamp)+ `preprocess_graph._process_one_graph`(timestep<time_steps 过滤节点与边)。
- **source-only 处理**: 单节点事件保留,空边表,depth=[0]。
- **source code locations**: 权威管线 `/data/jyz/next/src/rumor_detection/data`(本地镜像 `Algorithm/modified/ablation`,7 个核心文件中 4 个字节级一致;preprocess_graph/graph_builder/sentiment 三文件以服务器版为准)。完整 20 行字段表见 `feature_pipeline_audit.json`。

## Full-event Parity(seed=3090,100 事件,source-only/小/中/大 20/20/30/30,层内标签平衡)

| 检查项 | 结果 |
|---|---|
| node_ids exact | **100%** |
| num_nodes exact | **100%** |
| edge 结构一致 | 邻接签名 max_abs_diff = **0.0**(位级;最终 `.pt` 本就不存 edge_index 键) |
| node_feat[:, :384](text) | max_abs_diff **9.5e-7**(≤1e-5 通过;无需 cosine) |
| node_feat[:, 384:395](user 10 维) | **全部 0.0** |
| node_feat[:, 395](userFeatures,模型未用) | **0.0** |
| struct_feat[:, :1021](adjacency) | **0.0** |
| struct_feat[:, -3:](summary) | 30 分钟 bin 修正后 **0.0**(首次 60 分钟重建为 0.00208,系 bin 宽度参数差异,已定位) |
| 残留差异 | 仅 senti 列:max 0.068([0,1] 尺度)。**可完全解释**:当前 `sentiment.py` 以 fp16 autocast+batch64 推理,批序/精度噪声;fp32/batch1 对照差异 0.030,参考数据处于同一噪声带内。管线公式、文本、checkpoint 均一致 |

**结论:parity 差异完全可解释,不触发 `NOT_READY_PARITY`。** 附带发现:历史 240h 数据用 30 分钟时间 bin(`max_time_steps=480`),与当前管线默认(60 分钟)不同——重建 TC-DSCR 快照时必须显式固定 `time_window_minutes=30`。

## Candidate Snapshot Statistics

见 `pheme_snapshot_distribution.csv`(窗口 0/1/3/6/12/24/48h)。要点:

| 窗口 | mean nodes | median nodes | P90 nodes | mean edges | reply coverage P50 / P75 / P90 |
|---|---|---|---|---|---|
| 1h | 10.46 | 8 | 21 | 9.30 | 0.80 / 1.00 / 1.00 |
| 6h | 14.23 | 11 | 27 | 13.00 | 1.00 / 1.00 / 1.00 |
| 24h | 15.52 | 12 | 29 | 14.27 | 1.00 / 1.00 / 1.00 |
| 48h | 15.89 | 12 | 30 | 14.63 | 1.00 / 1.00 / 1.00 |

source-only 677 事件(10.54%)在所有窗口恒定;rumor 与 non-rumor 事件平均节点数差异小(1h:10.05 vs 10.71)。未自行决定正式窗口。

## Chronological Split Candidates

见 `pheme_chronological_split_candidates.csv`(60/20/20、70/10/20、70/15/15,按 source UTC 排序):

| split | train rumor 比 | validation rumor 比 | test rumor 比 | 跨段 topic 数 |
|---|---|---|---|---|
| 60/20/20 | 43.4% | **12.1%** | 44.7% | 1 / 9 |
| 70/10/20 | 38.5% | **14.9%** | 44.7% | 1 / 9 |
| 70/15/15 | 38.5% | **24.2%** | 45.3% | 1 / 9 |

两个结构性风险:(a) validation 的 rumor 比例(12–24%)远低于全局 37.4%,类别严重失衡;(b) PHEME 事件时间高度聚集于 9 个话题,**9 个 topic 中仅 1 个跨段**——chronological split 实际近似 topic split(跨话题泛化设定),且 validation 时间跨度仅约 2 天(2015-01-07~09)。按协议只报告,不改 random split。

## Chinese Semantic Compatibility

- 编码器本身无碍:历史 PHEME 与 Ma-Weibo 预处理使用**同一个** `paraphrase-multilingual-MiniLM-L12-v2`(多语言,官方支持中文),权重在服务器 `/data/jyz/rumor_detection/model/…`,tokenizer 可冻结;P4 以当前环境复现 384D 输出差 ≤9.5e-7,证明版本可锁定。
- Weibo22 因无原文不可编码(词 index 无法还原文本)。**不触发 `NEED_TEXT_ENCODER_DECISION`**(该标记针对编码器能力;问题在数据无文本)。
- 注意:Ma-Weibo 现有 raw(JSON,含 `t` 时间戳与原文)是潜在的中文替代来源,未经授权不在本轮展开。

## Blocking Issues

1. **Weibo22 无时间戳/无原文/无用户字段** → `NOT_READY_WEIBO22_TEMPORAL`:该 release 无法支撑 TC-DSCR 的第二数据集(多时间快照、chronological split、384D/11D 全部不可构造)。
2. **PHEME parent coverage 98.97% < 99.9% 预注册门槛**(1,017 条回复无 parent 字段;1.76% 回复 parent 指向事件外)。现有管线按孤立点处理,重建不受阻,但门槛未过需审批处置。
3. **senti 维未达 1e-5 严格 parity**(max 0.068):fp16 推理批序噪声;fp32/batch1 对照 0.030;其余全部特征位级或 ≤1e-5。需要审批接受"推理噪声带 + 预注册 fp32 推理"的处置。
4. **chronological split 的类别失衡与 topic 隔离**(validation rumor 12–24%;9 个 topic 仅 1 个跨段):影响"严格 chronological"评估的可解释性。

## Questions Requiring Research Approval

1. **第二数据集处置**:Weibo22 判 NOT_READY 后,TC-DSCR 是否改为 PHEME 单数据集先行,还是审计 Ma-Weibo raw(已有 `t` 时间戳与原文、且是 UMER 训练过的分布)作为中文侧替代?(后者需新一轮 preflight 审计其传播树方向性与 11D 用户字段可用性。)
2. **norm_degree 的快照定义**:其分母是事件内最大 raw_degree(依赖未来节点)。快照重算只能选:(a) 快照内 max(因果、值随快照变化)或 (b) 数据集级冻结常量。需批准其一并写入 causal snapshot feature builder 规范。
3. **senti 推理噪声处置**:是否接受 ≤0.07 的推理噪声带,并在 TC-DSCR 中预注册 fp32/batch 固定推理以获得可复现 senti?
4. **chronological split 的接受标准**:是否接受"近似 topic split + validation 类别失衡"作为时间泛化评估的代价,或需要设计 hybrid(如 topic 内 chronological)?
5. **PHEME 无 parent 回复的快照语义**:保持 UMER 现状(孤立节点)还是为 TC-DSCR 定义"默认挂到 source"的新规则(需防止拓扑泄漏)?

## Files Produced

```text
results/tcdscr/preflight/
  environment.json                     # git SHA / 运行环境 / 数据与代码路径(P0)
  environment_server.json              # 服务器侧环境探针原始输出
  repo_inventory.json                  # 仓库结构清单(P0)
  pt_probe.json                        # 处理图 .pt 键/维度探针(396+1024 维;node_ids/time_bin/max_time_steps)
  pheme_audit.json                     # P2 全量时间/父子/时序审计 + 门槛判定
  pheme_snapshot_distribution.csv      # P5 七窗口统计
  pheme_chronological_split_candidates.csv  # P6 三候选切分
  feature_pipeline_audit.json          # P1 20 行特征字段表(含 depends_on_future_graph / recomputable)
  full_parity_report.json              # P4 100 事件 parity 逐事件结果与汇总
  p4_diagnostic.json                   # P4 差异列定位(全部差异→senti 列)
  p4b_sentiment_control.json           # senti fp32/batch1 对照(归一化对齐)
  text_encoder_11d_audit.json          # P7 编码器审计 + P8 Weibo22 11D 逐维表
  snapshot_prototype.json              # P9 10 事件×3 截点原型机检结果
  manual_snapshot_examples.md          # P9 3 事件人工可读快照样例
  p2_run.log / p4_run.log              # 服务器运行日志
```

服务器侧同目录另存:`pheme_event_records.jsonl`(6,425 行事件级派生审计记录,含时间/父子拓扑,按"原始数据不入库"规则仅存服务器)、`p4_rebuild/`(重建中间产物)与全部执行脚本。
