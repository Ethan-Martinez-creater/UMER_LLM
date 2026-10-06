# SES-v1 P0-R1：审查修复与有限证据补齐计划

- 日期：2026-10-06。
- 审查对象：`212eacdbb4bd851b25abc1c8b223a926808e84cc`。
- [被审查分支](https://github.com/Ethan-Martinez-creater/UMER_LLM/tree/research/ses-v1-p0-20261006)。
- 本轮审批：**P0 交付暂不通过，需修复；`SES_P0_NOT_READY` 总判决保留；不批准 P1 标注或 reader 实验。**
- 下一轮：`P0-R1`，不是 P1；只修复审计、代码和设计草案，并在明确预算内核查缺失证据。
- 原 [P0 计划](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/docs/check/SES_V1_P0_DATA_AND_NOVELTY_ADMISSION_PLAN_2026-10-06.md) 的约束仍适用；本计划只明确修复项和有限扩展额度，不降低准入门槛。

## 1. 独立执行规范

**具体执行规范和要求需要从 UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md 中检索。**

遵守简体中文、PowerShell 7+、LOCAL／SERVER、模型／环境、历史结果只读、数据许可、人工确认规则。交接首页 M1-E 不代表最新成功；BCR／CR-TSER 保持关闭。缺凭据或授权不得用假值继续。

检索限定本项目 `docs/check/ses_v1/`、`scripts/ses_v1/p0/`、`tests/ses_v1/p0/`、清单已定位的 `local_assets/ses_v1/p0/`；不得扩大到盘根、用户目录、缓存或服务器全树。递归命令限时 60 秒，候选最多 50、命中最多 20，超限分批收窄。

不加载模型、不运行 GPU／reader、不训练、不生成攻击、不招募标注者、不安装依赖、不写 C 盘、不修改历史实验资产。原始正文、个人映射、网页快照不入 Git。代码测试 fixture 仅验证程序，不代替科研数据。

## 2. 审查已确认的有效工作

通过只读远端核验：

- 被审查分支为 `212eacd`；远端 main 为其父提交 `693eeed724d2f4391463b1150a1ab66467f82e61`，没有本轮更新 main 的证据。
- 17 个新增文件均属本轮报告、计划和 CPU 审计／测试，未见新增模型权重或原始包。
- 重新运行六项提交单元测试，6/6 通过；不是只采用执行者自述。
- 实际包 SHA256 为 `079f6ffdbc0b367399262f101774372e5d19dd8278c33d6c97a84461a9bc58dd`，与报告一致。
- 从本地真实 rv_map 重放固定抽样，与远端 manifest 一致；60 条原始 source/reaction 的 15/60/360 分钟计数与 manifest 一致，15 条在 15 分钟无候选。
- 实际包 275248 成员，无符号链接、硬链接或特殊文件；本轮不把一般归档风险当成已发生安全事件。

这支持工程基础，不支持配对有效性或科研机制。主智能体未执行写输出的完整 verify_p0，也未重新运行 compileall；仅重跑无写入单测与独立只读检查。

## 3. 必须修复的问题（按影响排序）

### F1 [P1] 取消“五条严格强配对，仅差一条”的结论

定位：[P0_CASE_AUDIT.md:97–107](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/docs/check/ses_v1/p0/P0_CASE_AUDIT.md#L97-L107)、`P0_VERDICT.json:51–54`。

现有记录不足以确认“同源复述＋有效独立纠错”。主智能体核查原始输入后：

- **007**：根命题是 144 位乘客＋6 位机组及坠机；R008 的国籍更正不反驳此命题，R012 重述同样人数且仍引 BBC。不能将侧面属性更正或更新转述当作根命题独立反证。
- **014**：R017 对 R010 的回复引用 CBC／警方确认，而复述也来自 CBC。该过程可作归属质疑／确认案例，不证明新的独立上游或根命题纠错。
- **028**：R004 是 R003 的回复，引用同一检方纠正信息；它是纠错传播，不是新增独立事实支持。引用网页历史版本仍未知。
- **026**：无链接否定和来源质询是信号，不因最终 V=false 而成为有效证据；根源帖含空域、歌剧院、旗帜多个命题，应分开定位。
- **053**：所引经理否认缺材料，其与根主张对应及上游独立性尚未核实；不能仅因有 deny 词而列为有效纠错。

分别保留探索候选、侧命题更正、来源确认、相关反驳信号和未知记录。严格合格数量应由逐项证据重算；目前不接受已确认 5 条的表述，也不把 UNKNOWN 自动当 FAIL 或 NO_GO。不得以“补一条”作为本轮目标。

### F2 [P1] 修复发现／确认集污染及剩余池数量

定位：[P0_NEXT_STAGE_DRAFT.md:8、23–25](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/docs/check/ses_v1/p0/P0_NEXT_STAGE_DRAFT.md#L8)。

草案将新增深查线程一方面列为发现资产，另一方面列为确认集。凡正文、纠错或结果用于设计的线程及其相关同主张线程不得再作为最终确认集。剩余线程应是 `2402−60=2342`，不是 2184；若存在别的排除项须逐项列分母，不能直接写未经解释的池。

本轮不从剩余池另抽 12 条。所有 P0 和 P0-R1 深查仍限固定 60；确认池只按尚未审读的同主张／话题组规则提出隔离方案，不读取确认正文、不冻结一个已经调过设计的假确认集。

### F3 [P1] 移除部署输入中的事后 annotation 与虚假独立性

定位：[P0_NEAREST_WORKS.md:35–45](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/docs/check/ses_v1/p0/P0_NEAREST_WORKS.md#L35)。

H1 将 annotation links.position 纳入输入，H2 将子回复确认称为独立响应；这与只使用当时材料、回复关系不等于来源独立的要求冲突。

annotation 的 FOR／AGAINST／observing 只能用于离线核查定位，未经当时可用性证明不得作为可部署输入、筛选权重、有效纠错标识或推理特征；annotation 本身绝不能进入 reader。

同一纠错的子回复可能依赖它，不能给一条真实反证凭多次附和追加独立权重。H1 目前的 URL／RT 分组计数与自身最强规则去重基线未显示实质算法差异；H2 “只改选择与权重、不改 prompt”也不足以证明创新。具体剩余机制没有证据时，将 `distinct_falsifiable_hypothesis` 降为 `UNRESOLVED`，不要为过关编架构。

### F4 [P2] 修复观察表字段语义及验证器假阳性

定位：[P0_OBSERVABILITY.csv:15](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/docs/check/ses_v1/p0/P0_OBSERVABILITY.csv#L15)、另有行 37、54；[verify_p0.py:104–116](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/scripts/ses_v1/p0/verify_p0.py#L104-L116)。

Python csv.DictReader 与 PowerShell ConvertFrom-Csv 均重现：014 的 correction_candidate 被正文占据，correction_evidence 变成 before_cutoff_verifiable，temporal_availability 变成 CORE PAIR CANDIDATE 文本；036／053 的类别字段也有非状态内容。所有行列数均等、provisional=1，现有验证器仍判全通过，说明仅列数／字符串检查不足。

改为显式字段写 CSV，不手拼字符串；采用固定枚举、键集合、数据类型、唯一性和状态／证据位置检查。unknown 不因非空就等同 yes。报告中的所有数量从结构化记录计算，不能手写汇总。

### F5 [P2] 拆分回复出现时间与引用材料历史内容

定位：[P0_OBSERVABILITY.csv:8、27–29、54](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/docs/check/ses_v1/p0/P0_OBSERVABILITY.csv#L8)、`P0_CASE_AUDIT.md:25–28、51、59、83–86`。

观察表标 before_cutoff_verifiable，案例正文却称历史网页未知，验证器只查两个词是否共现且不能阻断这种错误。回复创建时间可核，不代表它所引材料内容可核。全线程单一时间状态也不能表达 15/60/360 各截点。

还存在 001 notes 将 21.4–22.1 分钟 URL 簇称为 15 分钟内的错误。按真实逐行／逐材料时间与内容证据重算。截点后存档不能单独证明截点时文本；晚存档、当前页面和 annotation 回顾性链接分别保留。

### F6 [P2] 补齐真实字段覆盖、父边异常与输入哈希

定位：[build_rv_map_and_sample.py:226–257](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/scripts/ses_v1/p0/build_rv_map_and_sample.py#L226)、`P0_ASSET_AUDIT.json`、`P0_REPORT.md:14`。

资产报告缺版本／读取位置、作者／parent／URL／tree 覆盖和逐线程内容哈希；manifest 的 thread_hash 是 ID 伪名，不是输入内容哈希。代码定义 n_parent_conflict 却未检查，不能据此写“完整父子链”。

主智能体对固定 60 的只读检查：source/reaction 时间均可解析、无负时间差；36 个 reaction 在 structure 映射中缺父边，其中 23 个有 metadata parent；两者都有边时未见父节点不一致；未见父时间晚于子时间。这是覆盖缺口，不可统称 23 个真实冲突，也不可默认补成 SRC。

统一证据 ID：signal_summary 的 R### 按文件排序，deep_extract 的 R### 按时间排序，不能直接联结。采用本地稳定原始行映射和公开匿名 evidence_id，时间相同按固定稳定键排序。深查支持空 list 叶、缺 parent、缺 timestamp，避免全体 60 扩展时出错。

### F7 [P2] 收窄近邻排除结论并补方法差异证据

定位：[P0_NEAREST_WORKS.md:15、28–30](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/212eacdbb4bd851b25abc1c8b223a926808e84cc/docs/check/ses_v1/p0/P0_NEAREST_WORKS.md#L15)、`P0_REPORT.md:20`。

IMRRF 仅摘要／仓库核查，却被断言“冗余≈不相关干扰”“未区分四概念”；不能从有限阅读推出缺失。LLM-TKT／GAVEL 保留未解，同时重新检查 IMRRF。表格核查级别与“4 正文／2 摘要／1 元数据”汇总不一致，应程序化统计可核事实，不把无同时覆盖写成穷尽排除。

WFW 的 source preference 不能称已经证明事实真假准确率受同源重复影响。真实线程＋两个要求的组合不自动产生新方法；若剩余算法仍是现成规则去重，应明确创新未成立。

## 4. P0-R1 固定工作顺序与预算

### A. 先修复数据结构、计算与验证（不依赖外部全文）

1. 在同一研究分支追加提交，记录被审查 SHA／实际 HEAD／数据版本和读取位置。
2. 原固定 manifest 的 60 个 ID、选样顺序、R/V 和数据版本保持不变；新增输入文件 SHA256 等列，不改盐、不替换无信号样本。
3. 在审读新线程前修复 F1–F7 的已有结论，按下述状态重新编码；逐项重算数量和准入条件。
4. 完成真正 schema、时间及语义回归测试，不能只为得到 all_pass 改报告标签。
5. 尽量在合法可发布范围内保留公开证据定位；原文与个人 ID 保持本地。

### B. 有限历史证据核查

仅核查现有 12 深查例及 C 项新增最多 12 例所引的明确 URL／命题出处，禁止全网扩大搜索寻找“成功配对”。

- 本轮新增引用资源最多 18 个不同页面／快照对象；访问、重定向、存档查询及重试的实际 HTTP 尝试合计最多 48 次，每请求有超时。两种预算独立记账，失败也计入，不能通过改 URL／工具逃避额度。
- 先检查本地已有 P0 请求／内容哈希，再合法查询原短链及已证 canonical URL 的存档、publisher 固定历史页面；不因当前短链无存档就断言原长链也无。
- 请求账本记录 requested_url、final_url、status、timestamp、content_sha256、相关匿名 evidence_id、快照时间、相对截点、结果类别。无可读内容也留错误类型，不假装成功。
- 只有准确绑定命题及当时版本的材料才可支撑有效性。当前记者／账号的可信名声、最终 V、晚存档和后验文献不代替当时内容。
- 后续人工标注不能恢复不存在的历史证据；找不到就 UNKNOWN。登录墙、付费和访问控制不绕过。
- 限制达到或已确认关键材料不可补时停止网络工作，继续本地报告，不无限重试。

### C. 仅在原固定 60 内增查最多 12 条（明确的小额度扩展）

这是相对 P0 的唯一线程深查扩展，最多由 12 增到 24，不另抽剩余 2342 条。

对每个 V 类别，排除原 12 个深查 ID，按冻结 manifest 顺序取最早 4 条，不按信号、作者或“可能成功”再筛。先输出带哈希的新增发现清单，再读正文；原有无信号例继续保留。12 条全部报告结果，不能替换。

仍为 provisional 观察，不做人工 gold／一致性／模型判断。新增内容仅支持发现可标注性，不支持正式效应或成功率估计。原 60 及同主张相关线程的隔离风险须记录。

### D. 有限近邻核查

重点补 LLM-TKT、GAVEL、IMRRF 官方／作者全文，复核 WFW、Speaker-Free Floor 中与本方案直接相关的节号。既有 9 篇以字段方式逐一记录，不要求重复读已明确的无关部分。

H1 是分组计数／聚合，补查 RobustRAG 更新全文作为第 10 篇，区分文档隔离聚合与原始证据依赖推断；不能仅因不叫“认证”便略过。新增最多 2 篇最直接方法近邻，总数不超过原计划 12。

全文只走合法公开入口，原文保存在本地新子目录，记录版本／哈希。关键点仍不可判时 UNRESOLVED，不因“又试了一次”降低准入。

## 5. 结构化修复规范

新增 `P0R1_EVIDENCE_LEDGER.csv` 与 `P0R1_PAIR_AUDIT.csv`，不要用散文中 CORE PAIR 的关键词决定 gate。

每份材料／回复至少记录：匿名 case/evidence_id、明确 target_claim_id、是根命题／侧命题／上下文、provisional、source_kind（reaction／source／historical_external／annotation_audit_only）、reply_offset、cutoff、tweet_presence、historical_content_state、attribution_state、dependency_state、support_state、evidence_ref、未知原因。

固定状态须显式枚举，例如：

- `tweet_presence`：BEFORE_CUTOFF／AFTER_CUTOFF／MISSING_TIME／NOT_A_TWEET。
- `historical_content_state`：VERIFIED_AT_CUTOFF／AFTER_CUTOFF_ONLY／CURRENT_ONLY／UNKNOWN／NOT_APPLICABLE。
- `dependency_state`：VERIFIED_COMMON_UPSTREAM／VERIFIED_DISTINCT_UPSTREAM_CANDIDATE／UNKNOWN／NOT_APPLICABLE。
- `support_state`：SUPPORTED_CORRECTION_CANDIDATE／ASSERTION_ONLY／QUESTION_ONLY／CONFIRMATION_ONLY／SIDE_CLAIM_ONLY／INSUFFICIENT／UNKNOWN。

VERIFIED_DISTINCT 是有实际上游差异证据的候选，不能声称绝对独立。无法核实就 UNKNOWN，不通过各种缺省值升级。root 的真假与支持关系分开；多命题 source 不直接把整体 V 赋给每条子命题。

配对表每个 case/cutoff 列：共同上游证据、纠错针对的命题、纠错支撑证据、上游关系、历史可用性、是否满足原门槛、原因。计入严格配对必须全项可核，不能仅因有 deny／时间合格而通过；资格仍为 provisional，可交后续双人复核。

资产补齐：源码／tree／annotation／reaction 文件哈希、字段覆盖分母、缺时间、缺作者、parent 三种情况（tree 缺边、metadata 缺边、两者都有但不同）、缺 parent node、父晚于子、URL 原始字段及 query 是否影响内容身份。thread 伪名哈希不得改称内容哈希。

来源 query 不任意删除后就记相同内容；明确记录仅作 URL 候选聚类，不当已证共同事实来源。

## 6. 草案必须修复而非提前注册实验

- 全部已审读／设计用样本永久发现集；同主张／复制来源跨线程泄漏有具体隔离方案，统计不能把线程、主张、话题三个层级混称 event。
- annotation FOR／AGAINST、最终 V、人工审查结果、历史修复后才出现材料均从可部署输入剥离；用于评测的 oracle 属性单列。
- 新任务是 veracity／证据安全，不能直接沿用 A=RUMOR、B=NON_RUMOR 的旧任务 contract。以后可复用教师强制 likelihood 技术，但新标签、拒判动作、分数、校准和 prompt 都必须另审批；本轮不实现。
- 主指标以判决层错误风险、同覆盖率比较、有效纠错响应及反向误更新为基础。被选中某些行的比例只能作诊断，不叫事实错误支持率；true→wrong／false→correct 需明确指的是 reader 预测转移而非数据标签转移。
- 不增设“历史可核率 50%”等事后成功阈值；原六项准入不变。
- 保留 source-only、精确／语义去重、规则 URL/RT 去重、来源检查、重复不变方法和匹配覆盖率拒判等强基线，不能靠四臂草案遗漏最强对照。
- 人员仍为 PENDING_HUMAN_RESOURCE；本轮不实际标注，不以多模型一致替代人类核验。

## 7. 验证要求

必须增加并实际运行以下回归测试，测试合成 fixture 仅为程序验证：

1. 与 014 同类的字段错位／非枚举状态被拒绝，不能 all_pass。
2. reply 前于截点但网页 CURRENT_ONLY／UNKNOWN 时，配对不得通过；晚存档不得自动转 VERIFIED_AT_CUTOFF。
3. 侧命题更正、提问、同源确认、依赖子回复均不能计成独立有效根纠错。
4. annotation_audit_only 输入不能被送入部署候选；V 不决定纠错有效性。
5. 21.4 分钟材料不进入 15 分钟；每截点单独检查，不按散文关键词判断。
6. 双边缺失／一边缺失／两边不同、parent 晚于 child、空 tree list、缺／无效时间均有确定处理，不默默默认 SRC。
7. summary/deep 的证据 ID 联结一致，时间平局稳定；各原始输入哈希存在并匹配。
8. 已审读深查集与确认集交集为零，原盐 manifest 不变；剩余分母可重算。
9. 有意破坏 schema／时间／状态后验证器非零退出；所有失败被机器记录。
10. 报告配对数、维度状态、近邻核查级别与 ledger／verdict 一致；隐私检查不要仅匹配 17–19 位 ID，须人工检查待提交文本。

全量本地验证不改历史资产，验证脚本输出仅本轮派生文件。完整失败输出保留；检查未运行应写 NOT_RUN。禁止用测试通过覆盖证据 UNKNOWN。

## 8. 交付文件与停机判决

在原研究分支修复 `scripts/ses_v1/p0/`、`tests/ses_v1/p0/` 和 `docs/check/ses_v1/p0/` 中相应文件，保留 Git 审查历史；不修改原 R4／P0 计划以适应结果。

新增于 `docs/check/ses_v1/p0r1/`：

1. `P0R1_REVIEW_RESPONSE.md`：F1–F7 逐项回应、改动路径和验证证据。
2. `P0R1_EVIDENCE_LEDGER.csv`、`P0R1_PAIR_AUDIT.csv`：结构化证据与配对资格。
3. `P0R1_DISCOVERY_EXTENSION.csv`：在原 60 内预先固定的最多 12 个增查 ID，无新正文或原始个人 ID。
4. `P0R1_REQUEST_LEDGER.csv`：允许公开的请求元数据、哈希、预算结果；原文不入库。
5. `P0R1_NEAREST_WORKS.md`：合法版本、读取级别、节号、支持与未知、原假设与简单基线的实质差异。
6. `P0R1_VALIDATION.json`：包括回归测试及原 60 重放、原始字段、hash／schema／time 一致性。
7. `P0R1_VERDICT.json`：分别记录修复完成、原六项准入、配对数、模型 NOT_TESTED、gold PROVISIONAL_AUDIT_NOT_GOLD 和总状态。
8. `P0R1_REPORT.md`：不超过 1500 中文字符，说明哪些问题修复、哪些仍缺以及是否现实可补。

总判决可用：

- `SES_P0R1_READY_FOR_P1_DESIGN`：修复验证通过且原六项准入全满足；仍只送下一轮设计审批，不启动 P1。
- `SES_P0R1_NOT_READY`：修复完成，但数据／历史资料／全文／机制差异仍不足；明确缺项，结束本次补齐，不提无限增样／找存档。
- `SES_P0R1_NOVELTY_BLOCKED`：核实仅现成去重、来源提示、拒判拼装或直接近邻覆盖。
- `SES_P0R1_REPAIR_INCOMPLETE`：仍有本计划的可修复缺陷；数据缺失不能作为免修代码理由。
- `SES_P0R1_INFRASTRUCTURE_PAUSE`：真实凭据／权限／依赖阻塞；完成不依赖该项的工作并停下。

达到预算仍 NOT_READY 时交由主智能体决定是否关闭优先方向或转向备选，不自行降低门槛、换任务或自动启动下一轮。

## 9. Git 交付

在 `research/ses-v1-p0-20261006` 上基于 `212eacd` 追加普通提交；若远端已有后续提交先报告与对齐，不重置／force push、不改 main、不合并。保留用户既有工作树改动；逐路径暂存许可明确的派生报告、审计代码与测试。

本计划可一并提交，原始数据、日志正文、凭据、个人映射与模型不提交。发布权限未解部分只在本地保存并报告；缺认证则停发布，不伪造成功。

最终简报：修复结果、严格配对数量及分布、仍缺的证据、近邻与机制判定、测试结果、commit SHA、分支链接。**提交后停止，等待主智能体审批。**
