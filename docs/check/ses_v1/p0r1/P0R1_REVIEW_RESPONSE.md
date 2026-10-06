# P0-R1 审查回应：F1–F7 逐项修复记录

被审查提交：`212eacdbb4bd851b25abc1c8b223a926808e84cc`；本轮基于其追加提交，main 未动。
数据版本：figshare 6392078 v1，SHA256 `079f6ffd…9bc58dd`，读取位置 `local_assets/ses_v1/p0/extracted/all-rnr-annotated-threads`（本地固定）。

| 项 | 回应 | 改动路径与证据 |
|---|---|---|
| F1 | **接受。**"5 条强配对"撤销；24 例（原 12+增 12）配对资格由结构化证据重算：gate@15min = NO 21 / UNKNOWN 3（026、028、053）/ **YES 0**。026/028/053 保留为探索候选并记录缺项；UNKNOWN 不当 FAIL 也不当 PASS | `scripts/ses_v1/p0r1/build_ledger_and_pairs.py` → `P0R1_PAIR_AUDIT.csv`（72 行）、`P0R1_EVIDENCE_LEDGER.csv`（52 行） |
| F2 | **接受。**剩余池分母更正 2402−60=2342（2184 系笔误）；确认集改为"同主张/复制来源整组退出后抽样"方案，本轮不另抽、不读确认正文；统计层级 thread/claim/topic 三分 | `P0R1_NEXT_STAGE_DRAFT.md` §1 |
| F3 | **接受。**annotation/V/人工意见/晚截点存档列为不可部署输入；`schema_checks.check_ledger_schema` 强制 annotation 行不得为 SUPPORTED_CORRECTION_CANDIDATE；子回复确认降 CONFIRMATION_ONLY；H1/H2 降为探索候选，`distinct_falsifiable_hypothesis=UNRESOLVED` | `scripts/ses_v1/p0r1/schema_checks.py`；`P0R1_NEXT_STAGE_DRAFT.md` §2–3；`P0R1_VERDICT.json` cond6 |
| F4 | **接受。**OBSERVABILITY 全表 DictWriter 显式重写；correction_candidate 收敛为 {yes,no,unknown}（yes 0 / no 5 / unknown 55）；temporal_availability 收敛为 4 值枚举；verify_p0 升级枚举+状态/证据位置校验（原 all_pass 不可再被错位行通过——回归测试 1 覆盖） | `scripts/ses_v1/p0r1/fix_observability_r1.py`；`scripts/ses_v1/p0/verify_p0.py` §6b |
| F5 | **接受。**回复时间与材料历史内容拆分（historical_content_state 五值）；ledger 逐行按 offset 推导适用截点（001 的 21.4–22.1min URL 簇 = `60;360`，不再声称 15min 内）；历史核查（对象 1–3）确认三个 live/长链快照均晚于截点 → AFTER_CUTOFF_ONLY，"晚存档不升 VERIFIED_AT_CUTOFF"由回归测试 2 强制 | ledger `cutoff` 列；`P0R1_REQUEST_SUMMARY.json`；`tests/ses_v1/p0r1` |
| F6 | **接受。**60 条逐线程 4 类输入哈希齐备（零 reactions 线程允许空合并哈希）；字段覆盖 805 行、0 缺时间、0 缺作者；父边三态 tree_only=0 / metadata_only=23 / 双边不一致=0；树外 in_reply_to=17；父晚子=0——与主智能体独立检查一致，未默认补 SRC；伪名哈希改称 `thread_pseudonym_hash`；walk 支持空 list 叶并移至模块级 | `scripts/ses_v1/p0r1/asset_audit_r1.py` → `P0R1_ASSET_COVERAGE.json` + 本地 `asset_audit_full.json` |
| F7 | **接受。**IMRRF/GAVEL/RobustRAG 升级为正文（IMRRF"冗余"=§3.2 LLM 摘要式无关性过滤，推断获正文证实）；核查级别程序化统计：正文 9 / UNRESOLVED 1（LLM-TKT，已试 ACM/arXiv/Scholar/ResearchGate/GitHub/S2）；WFW 度量收窄为 source preference，记录 §6 缓解数字与口径差异 | `P0R1_NEAREST_WORKS.md` |

## 预算记账

- 历史核查：5 个对象（≤18）、5 次 HTTP（≤48），全部成功；账本 `P0R1_REQUEST_LEDGER.csv` + `P0R1_REQUEST_SUMMARY.json`（内容仅哈希，原文不入库）。
- 线程深查：12 → 24（上限内），增查清单先于阅读冻结（`P0R1_DISCOVERY_EXTENSION.csv`）。
- 近邻：10 篇（≤12），未新增其他近邻。

## 验证结果

- `tests/ses_v1/p0`：6 通过；`tests/ses_v1/p0r1`：16 通过（含 §7 十项对应断言；1 项 skip 为可选的官方脚本对比，因环境缺包跳过）。
- `verify_p0.py` all_pass（含新枚举校验）；`verify_p0r1.py` all_pass（重放、增查集合、ledger/pairs 枚举与语义、cutoff 一致、父边计数、哈希、预算、隐私、verdict 计数一致性）。
- compileall 通过。验证失败不隐藏：本轮曾出现并修复 `cutoff==15` 整型比较 bug（详见 Git 历史）。
