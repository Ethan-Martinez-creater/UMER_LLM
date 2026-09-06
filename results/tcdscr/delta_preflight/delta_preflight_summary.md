# TC-DSCR Delta Preflight Summary

**执行日期**:2026-09-06 · git commit `71ec7832…`(本轮验证提交后更新)· 服务器:`/data/jyz/next/llm/results/tcdscr/delta_preflight/`
**本轮性质**:第一轮审批后的增量验证;未训练模型、未改 UMER、未引入 11D、未自定窗口/split、未换数据集与 embedding。

## Overall Status

**PARTIAL** — PHEME 三项修正全部通过;Ma-Weibo 满足 TC-DSCR 第二数据集的全部最低条件(§20),384D + causal 1024D 在两个数据集上均可无泄漏构造;但存在需要研究审批方决定的 protocol 问题(Ma-Weibo chronological split 类别崩塌、正式窗口选择),故不判 PASS。

## PHEME Corrections

### P9 Time-bin Fix
- status: **PASS**
- definition: `time_bin = floor(elapsed_seconds/1800)` clip [0,479];`norm_time = time_bin/480`(与冻结历史定义 `max_time_steps=480` 一致;第一轮错误公式 `elapsed//3600/480` 已废弃,并在输出中保留对照)
- exact rate: **time_bin exact = 100%**(10 个冻结事件 × 1h/6h/24h,样本未更换);`norm_time max_abs_diff = 0`
- leakage: future topology leakage = **0**;future text leakage = **0**(快照特征仅由 included 节点重建);节点/边单调包含 **PASS**

### Parent Audit V2
- total replies: **98,929**(互斥三分,和恰为总数)
- resolved: **96,414(97.46%)**
- missing parent(字段为空): **1,017(1.03%)**,涉及 834 个事件
- external/unresolved(parent 不在本事件): **1,498(1.51%)**,涉及 634 个事件
- chronology violations: **0**
- 逐事件未解析比例 P50/P90/P99 = 0 / 8.3% / 36.4%;50 个异常样例已列出
- 处置按冻结规则:不人工挂到 source;后续 evidence packer 表示为 `[UNAVAILABLE]`(本轮未实现 packer)

### Early Snapshot Dynamics

| cutoff | source-only | median nodes | P90 nodes | median coverage | events with new nodes(vs 前一窗口) | new-edge events |
|---|---:|---:|---:|---:|---:|---:|
| 0m | 677 | 1 | 1 | 0.0% | — | — |
| 5m | 677 | 2 | 8 | 11.1% | **4,180** | 4,174 |
| 15m | 677 | 4 | 15 | 33.3% | **4,422** | 4,414 |
| 30m | 677 | 6 | 19 | 50.0% | 3,749 | 3,720 |
| 1h | 677 | 8 | 21 | 70.0% | 3,546 | 3,521 |
| 3h | 677 | 10 | 24 | 91.7% | 3,480 | 3,454 |
| 6h | 677 | 11 | 27 | 100% | 1,958 | 1,943 |
| 24h | 677 | 12 | 29 | 100% | 1,814 | 1,792 |

**5–30 分钟窗口存在大量动态**(每窗口 3,700–4,400 个事件有新节点与新边),支持 TC-DSCR 的 context refinement 设定;1h 中位覆盖率 70% 确认第一轮"1h 可能过晚"的判断。正式窗口不由本轮冻结。

## Ma-Weibo Raw Schema

- event count: **4,664**(labels 全部命中;380 万帖)
- raw text coverage: **99.9996%**(`original_text` 字段 100% 非空)
- time field: 每帖 `t`,int,**绝对 unix 秒**(全量范围 1,291,221,273–1,452,788,866,即 2010-11 ~ 2016-01);source 的 t 在 99.98% 事件中等于事件最小 t;负 elapsed 0;缺失 0
- time semantics: **ABSOLUTE_UNIX_SECONDS**(非 TIME_SEMANTICS_UNRESOLVED)
- parent relation: `parent` 字段(mid 字符串,null=根);**解析率 99.9992%**(仅 30 条 external),环 0,不可达 30,child<parent 仅 5;root 定义=parent==null(每事件恰 1 个,multi-root 0)
- within-event temporal ready: **yes**
- global chronological ready: **yes**(但见下文 split 结果)

## Ma-Weibo Feature Pipeline

- 384D encoder: `paraphrase-multilingual-MiniLM-L12-v2`(与 PHEME 同一模型;config `ma_weibo_240h.yaml`)
- **文本清洗发现(重要)**:冻结的 240h `.pt` 是对 **`clean_text_weibo(original_text)`** 编码的(d6_diagnostic:参考行与清洗文本 cosine = 1.0);当前服务器 `pipeline.py` 丢失了该步骤(版本漂移)。D6b 修复后恢复一致。
- 1021D adjacency: 与 PHEME 同规则(parent 行=出度归一+自环,child→parent,pad/截断 1021)
- 3D structural: norm_depth=depth/19(固定);norm_time=bin/480(**30 分钟 bin 与 PHEME 完全同定义,无需统一审批**);norm_degree 历史=事件内 max → 已按 V2 §2.3 冻结为快照内 max
- historical cutoff: 240h 过滤(elapsed 基准=事件最小 t)+ 30 分钟 bin(max_time_steps=480)
- authoritative source files: `/data/jyz/next/src/rumor_detection/data/{adapters/ma_weibo.py, ma_weibo/*, preprocess_graph.py, time_windows.py}`;完整 12 行字段表见 `maweibo_feature_pipeline_audit.json`

## Ma-Weibo Full Parity(seed=3090,100 事件:source-only 6 全取 + small/medium/large 层内标签平衡)

| 检查项 | 首次重建(未清洗) | **D6b(恢复清洗后)** |
|---|---|---|
| num_nodes exact | 100% | **100%** |
| time_bin 逐行 exact(节点级时序指纹) | 100% | **100%** |
| adjacency signature | 0.0(位级) | **0.0(位级)** |
| summary3 | 0.0(位级) | **0.0(位级)** |
| text384 max_abs_diff | 2.46(43/100 事件超 1e-5) | **1.43e-6(0/100 超阈值)** |
| extra11(11D,INFO 非通过条件) | 0.86 | 0.0048(senti 推理噪声带) |

参考 `.pt` 不含 `node_ids` 键(历史行为);节点顺序一致性由 num_nodes + time_bin 逐行 + text384 逐行 + 邻接位级共同证明。**未触发 `NOT_READY_MAWEIBO_FEATURE_PARITY`。**

## Ma-Weibo Snapshot Dynamics

见 `maweibo_snapshot_distribution.csv`(0m–24h 八窗口,含边与可达深度)。样例(10 事件原型,15m/1h/6h):事件 3475831861399909 在 15m 已有 7 节点 6 边(root 度 6),1h 82 节点 81 边;事件 3492448519351538 在 15m 13 节点、深度已到 4——**Ma-Weibo 早期传播比 PHEME 更深更快**,分钟级窗口的动态空间成立。rumor/non-rumor 平均节点数在各窗口差异小。

## Leakage Checks

- PHEME: 30min bin 修复后 P9 topology/text leakage = 0;节点/边单调 PASS
- Ma-Weibo: D8 原型 topology/text leakage = 0;单调 PASS;norm_degree 按 V2 §2.3 快照内独立归一化
- 执行过程中修复的两个审计脚本缺陷(int 索引与字符串节点集比较导致边计数为 0)已在合成样例自测后修正重跑;`d6_diagnostic.json` 保留首次差异的完整归因链

## Blocking Issues

无 NOT_READY 级 blocker。

## Questions Requiring Research Approval

1. **Ma-Weibo chronological split 类别崩塌**:三个时间切分候选的 validation/test **rumor 比例全部为 0**(谣言事件集中于 2012–2013,2015 年段全为非谣言;60/20/20 的 train rumor 82.7% vs val/test 0%)。技术上 global chronological 可行,但作为评估划分不可用。需审批:TC-DSCR 的 Ma-Weibo 评估是否改用"event-internal causal dynamic"为主(文档 §19 已预留此路径)并另行定义 split(如按年份分层的随机划分),或接受时间切分仅用于诊断报告。
2. **正式快照窗口**:PHEME 5–30 分钟动态显著;Ma-Weibo 15m 即达中位 36 节点、深度 3+。两数据集的正式窗口组合由审批方冻结(本轮不预设)。
3. **管线规范冻结**:TC-DSCR 正式实现应以"恢复清洗后的历史管线"为准(`clean_text_weibo(original_text)` → MiniLM;30 分钟 bin/480;norm_degree 快照内 max),并建议将该版本提交服务器 `src` 以消除版本漂移——需授权后执行。

## Recommended Status

**PARTIAL** — 建议审批方就上述 3 项 protocol 问题决策后,进入 Code Complete 阶段。
