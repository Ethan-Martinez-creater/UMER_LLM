# 下一轮建议计划：新方向准入审计（A0，CPU/只读）

日期：2026-10-03。基线：GitHub HEAD d71991dd9745a910e94ca91398d0b871a7ffda7f；BCR M1-F=CLOSE；M2未授权。

本计划等待用户选择后交执行智能体。只审计资源、标签与近邻，不开展正式模型实验。输出仅限 `docs/check/a0/`；历史代码、标签、结果和数据保持只读。

## 1. 独立执行规范

**具体执行规范和要求须先从 [UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md](../../UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md) 检索；远端 M1-F 的最新判决优先于该文档的旧阶段状态。**

- LOCAL只能实现/CPU审计，不能加载7B/8B模型。
- 如需查询SERVER数据，只通过交接文档规定连接入口；拒绝连接/缺凭据即停，不换host/port/方案。
- 不运行模型下载、pip/conda安装、reader inference、训练或GPU轮询；不修改全局环境。
- 不把 non-rumour 当 true，不把 unverified 当 false，不把最终标签当每个时间点的oracle。
- 不恢复CR-TSER/BCR，不重算旧cache、不改历史阈值/标签。
- 缺真实gold或许可，不用LLM、mock或默认值补齐。

## 2. A0-A：现有PHEME双标签资产审计

### 操作

1. 核验当前目录与本地/远端commit，记录差异；只读Git命令，不checkout/pull/reset。
2. 按交接文档定位原始PHEME与现有processed/manifest/utility输入，列出实际可读资产及SHA256。不假设公开仓库包含原始大数据。
3. 建立 `thread_id -> news_topic -> source_hash -> R -> V -> 原始标注路径` 映射。R仅来自原始rumours/non-rumours来源；V仅来自真实annotation及数据集原始定义，不从文本或目录猜真假。
4. 只统计映射和覆盖率，不改写任何原始数据。R=non-rumour缺V时记not_applicable，不补true。
5. 报告每话题 true-rumour / false-rumour / unverified-rumour / non-rumour 数量、缺失/冲突/重复ID、与processed训练标签和BCR prompt的对应关系。人工spot-check最多20个匿名ID+哈希，不复制整段用户原文进报告。
6. 核查当前reader prompt究竟定义R还是V；报告中区分“题面有歧义”“直接错任务”“数据映射一致”。

### 探索pilot准入线（本轮拟定，不是发表标准）

- 双标签映射：可读R的thread与processed对应覆盖率≥95%，R冲突未解决数=0。
- 各 V 子集 usable thread≥50，分布至少3个新闻话题；至少1个可独立留出的新闻话题同时含true/false rumours且各≥10。若未达，报告实际数，判样本/划分受限，不临时放宽。
- 原始同一事件/近重复跨划分风险有可执行阻断方案。
- 至少一个项目外系统具有可核查的公开输入/prompt/标签/实现映射，并指出待检验的任务错位假设；只有内部M1-F证据不判新候选过关。缺此项记独立贡献待核验，不进入pilot评审。
- 程序保留R/V独立字段；没有拟把V缺失自动当false/abstain的设计。

上述只决定是否足以做发现性pilot，不构成统计功效保证；正式pilot前再按最小效应/不确定性制定样本量。任何一个关键项不满足，A0-A=`NOT_READY`；不以“先跑起来再说”进入GPU。

## 3. A0-B：CoPHEME可用性与真值证书审计

### 操作

1. 固定作者公开仓库commit，核验README、LICENSE/数据许可、六个事件JSON、公开plans、loader、runner；不执行作者demo，不配置API key。
2. 复核本轮402 false /688 unverified目标及3942 non-rumour /1008 true证据行数；差异报告版本和路径，不修改旧记录。
3. 定位公开runner的目标/证据过滤，记录plans与target ID、原始source ID、提取conclusion的绑定。另查公开配置/样本ID是否与论文eval有可核查对应；无法绑定时只称“公开runner可重建集合”，论文实际eval集保持UNKNOWN，不把公开池等同论文样本。
4. 按 `SHA256(topic|target_id)` 排序，选前20个V=false目标及前20个V=unverified目标（数量不足则全取）；检查是否提供conclusion真假证书、片段真值证书、合法多跳正例、原始文本/时间链接。最多读与这40个目标对应的plans，不生成新攻击。
5. 实际存在的证书逐项列出；仅原始source veracity或Director打分，不记为片段/新conclusion的事实gold。
6. 证据与conclusion真值未证实、许可未明确、或没有正常多跳正例时，判防御pilot `NOT_READY`。本轮不自行找人标注或替作者核实全部事实。

### 准入线

分别判定两类gold，不可互替：

- 若研究“真片段→假结论”及其攻击成功率，须有片段与目标conclusion的命题级事实真假gold，另有支持链/可推出性gold及正常多跳对照。
- 若只有可核查支持/非支持标注，只能研究限定证据集内的unsupported inference，不得称结论为假或所有片段为真；候选应按此收窄并重新查重。

只有使用/必要时发布派生统计的权限明确、所选任务gold完整、且正常多跳对照可得，才建议下一阶段防御pilot。当前代码可访问≠许可明确，non-rumour≠LT=1，false source≠extracted conclusion必假，non-support≠false。若gold单位与监督退化为旧跨事件纠正关系，沿用旧一致性失败约束，不通过换名立项。

## 4. A0-C：针对两个候选做最后一轮方法级排重

- A0-A候选：查LLM rumour-vs-veracity任务错位、双标签评测、true-rumour子集、跨语言目标混淆；至少核查6个最近邻全文或官方页，其中必须包含TripleFact、PHEME原始定义、M1-F。
- A0-B候选：查Generative Montage第6节/后续引用、GRIL、GraphEcho、CAG/CONFACT、support-chain/unsupported inference defense；至少核查6个最近邻。
- 每篇表格列 `已解决问题 / 实验设定 / 我们新增问题 / 仅换域还是新增机制 / 核查级别 / URL`。
- 找到已发表的实质相同问题与解法，即该候选=`NOVELTY_BLOCKED`。仅没有同名论文不记PASS。
- 不把未能读全文记“没有做”；记录UNCERTAIN；关键uncertain未消除时不正式立项。

## 5. 最终允许的结果

- `A0_TASK_VALIDITY_PILOT_REVIEW`：仅A0-A资产和候选问题过关，申请下一轮**独立**prompt/模型pilot协议；不自动运行。
- `A0_DEFENSE_PILOT_REVIEW`：A0-B资产/许可/gold与区别过关，申请独立防御pilot协议；不自动运行。
- `A0_BOTH_FOR_REVIEW`：两者过关，但仍只送研究评审，由用户/指挥选择，禁止并发扩大。
- `A0_NO_ADMISSIBLE_DIRECTION`：关键资源/新颖性不足，不生成一个换名方向。
- `INFRASTRUCTURE_PAUSE`：数据连接受阻，写精确错误与缺失项，不把它称科研假设失败。

## 6. 固定输出

均在 `docs/check/a0/`：

1. `ASSET_AUDIT.json`：commit、资产路径/哈希、R/V统计、覆盖/冲突/重复、SERVER是否访问；不包含凭据。
2. `PHEME_TASK_MAP.csv`：thread_id/topic/source_hash/R/V/annotation出处；如许可不允许分享ID，仅保留本地并在报告说明，不推送未经许可的逐样本清单。
3. `COPHEME_GOLD_AUDIT.md`：样本、真实证书字段、过滤逻辑、许可、限制；不发布用户原文。
4. `NEAREST_WORKS.md`：两候选逐篇方法对照与核查级别。
5. `A0_VERDICT.json`：每项gate与具体证据，不将NOT_READY写FAIL。
6. `A0_REPORT.md`：≤1500中文字符的决策摘要，链接完整表格，结尾确认无推理/训练/新标签/历史修改。

本轮不得创建reader推理脚本、训练器或新模型。审计后STOP，等审批；Git提交/推送依用户当轮授权及数据许可执行，不为发布而越过许可硬门。

## 7. 可直接给执行智能体的短提示词

> 先从 [UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md](../../UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md) 检索具体执行规范和要求；阶段状态以远端M1-F=CLOSE为准。
>
> 按本A0计划，仅做PHEME原生R/V双标签映射、CoPHEME过滤/gold/许可审计和两候选方法排重。核验commit与资产哈希，所有新输出写docs/check/a0。禁止模型加载、GPU推理、训练、生成新标签、修改历史资产和自动进入下一阶段。逐项gate写证据；缺资源报NOT_READY/精确错误，不用猜测补齐。提交审计报告后STOP等审批。
