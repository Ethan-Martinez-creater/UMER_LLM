# TC-DSCR Delta Preflight(V2 第二轮验证)结果

执行日期:2026-09-06 · 脚本:`scripts/tcdscr_delta_preflight/` · 服务器工作目录:`/data/jyz/next/llm/results/tcdscr/delta_preflight/`

| 文件 | 步骤 | 内容 |
|---|---|---|
| `environment_delta.json` | D0 | 环境/版本/路径冻结,第一轮结果完整性确认 |
| `pheme_p9_fixed.json` | D1 | 30 分钟 time-bin 修复后的 P9 原型(同 10 事件/seed 3090/1h/6h/24h) |
| `pheme_parent_audit_v2.json` | D2 | parent 三类互斥审计(resolved/missing/external)+ 50 例 |
| `pheme_early_snapshot_distribution.csv` `_summary.json` | D3 | 0m/5m/15m/30m/1h/3h/6h/24h 窗口统计 + 相邻窗口动态 |
| `maweibo_raw_schema_audit.json` | D4.1 | 4664 事件/380 万帖全量字段覆盖 + t 语义判定 |
| `maweibo_temporal_audit.json` | D4.2/4.3 | 时间覆盖/负值/重复 + 传播关系(解析率/环/可达性) |
| `maweibo_snapshot_distribution.csv` | D7 | 与 PHEME 同口径的 8 窗口分布(含边与深度) |
| `maweibo_feature_pipeline_audit.json` | D5 | 12 行字段表(权威管线定位) |
| `maweibo_full_parity_report.json` | D6 | 100 事件 seed 3090 全量重建对比 |
| `maweibo_snapshot_prototype.json` `_examples.md` | D8 | 15m/1h/6h 因果快照原型(10 事件) |
| `maweibo_chronological_split_candidates.json` | D9 | 绝对时间三候选切分 |
| `delta_preflight_summary.md` | — | 最终汇总(模板 §22) |

服务器侧另有(不入库):`maweibo_event_source_times.jsonl`(逐事件源帖绝对时间)、`d6_rebuild/`(重建中间产物)、各 `*_run.log` 与全部执行脚本副本。

结论以 `delta_preflight_summary.md` 为准;原始数据未提交仓库。
