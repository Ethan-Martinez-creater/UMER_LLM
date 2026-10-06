# SES-v1 P0 深入案例审计（12 条）

- 审查者：执行智能体（provisional 审查笔记，**非人工 gold**，无双评一致性）。
- 对象：固定 60 条中的 12 条（按 manifest 顺序、具备所需材料者），深查视图由 `scripts/ses_v1/p0/deep_extract.py` 从原始 JSON 生成。
- 定位方式：`R###` 为时间排序后的匿名行号；`@Xmin` 为相对源帖的分钟偏移；`SRC` 为源帖；`ANN-LINK` 为 PHEME 原生 annotation.json 的链接（含原生 `position`（for/against/observing）与 `mediatype` 字段）。
- 隐私：不复制用户原文整段；只保留必要短引文（≤25 词）；不记录任何用户标识。
- 公开页面核实额度：9 次请求（≤12 上限），逐条记录见各案例"页面核实"。

## 通用发现（跨案例）

1. **PHEME veracity 包的 annotation.json 原生携带链接级 for/against/observing 标注**（本项目此前未使用）。这为"来源归属"与"立场"提供了非 LLM 的原生锚点，但它是**标注者对 thread 的整体归档**，不是逐回复、逐截点的依赖标注。
2. **历史截点可用性普遍不可核**：两个 bbc.in 短链现在重定向到 live 页面（内容持续更新）；cbc.ca 链接当前指向"受害者确认"页面而非 RT 时刻的疏散报道——实证了"当前可读 ≠ 截点时内容"。短链本身在 Wayback 无存档。
3. 树结构（structure.json）+ reaction 的 in_reply_to 字段可以恢复完整父子链；深查 12 条中树深最大达 17 层（SES-P0-056），离题对话是真实线程的常态。
4. 有效纠错在树内**通常以三种形态出现**：(a) 无链接的断言式纠错（快但弱）；(b) 带链接的引用式纠错（强但少）；(c) 对他人纠错的树内确认/传播（如 028-R004 回复 R003）。

## 逐案例

### 1. SES-P0-001 [true] germanwings-crash
- 同源复述：R002/R010 逐字 RT @flightradar24（源帖本身即 flightradar24 官方）；R008/R014 引述同一片段。
- 依赖证据：R024-R028 五个不同账号在 21.4-22.1min 共享同一 blogspot URL（末日预言博客，与事件无关的可疑材料）。
- 纠错：无。R005 @3.85min 引用官方确认（laprovence 链接）。
- 页面核实：无（额度用于其他案例）。

### 2. SES-P0-007 [true] germanwings-crash ★配对候选
- 同源复述：**两个独立 BBC URL 簇**——bbc.in/1EMTxMz 被 4 条共享（R002/R003/R012/R017 @3.0-67.0min）；bbc.in/1LSgxjV 被 5 条共享（R004/R005/R006/R010/R011 @7.1-27.5min）。
- 纠错候选：R008 @14.38min 要求更正国籍信息；R012 @28.37min 日语"訂正"（人数更正）——两者都指向同一 BBC 材料的不同时点版本。
- 树：最深 5 层；R004-R006 聚在同一个二级节点下。
- 页面核实：bbc.in/1EMTxMz Wayback 无存档；当前重定向到 live 页（内容不可回溯）→ `TEMPORAL_AVAILABILITY_UNKNOWN`。

### 3. SES-P0-008 [true] ottawashooting
- 同源复述：R001-R019 大量同文 RT @CNN（0.8-179.6min），无 URL 聚类（reactions 未展开链接）。
- 纠错：无；纯回声簇 + 少量反应。

### 4. SES-P0-010 [true] ferguson
- 同源复述：R001 @0.0min 与源帖完全相同（转发链顶点）。
- 树内大量政治争论（88 条，深度多层）；R006 @7.8min 跨来源引用（"Al Jazeera America just reported too"）。
- 质疑：R078 @380.75min 质疑视频造假——**在 360min 截点后**（AFTER_CUTOFF）。
- 页面核实：无。

### 5. SES-P0-014 [true] ottawashooting ★配对候选
- 来源归属质疑→确认微过程：R008 @6.27min 质询"来源与出处是否核实"；R010 @7.22min 指出照片此前被归于某极端组织账号；R017 @10.75min 树内回应"警方消息源已向 CBC 确认"；R024 @53min 追溯传闻链（"人们声称从该账号获得"）。
- 同源复述：R001/R006/R015/R016 引述 @CBCOttawa。
- 树：深达 9 层，大量离题对话（语言争论），离题噪声显著。
- 启示：归属质疑与确认在同一棵树内 10 分钟内完成交锋——依赖感知机制的理想微观案例。

### 6. SES-P0-026 [false] sydneysiege ★核心配对候选
- **R001 @0.4min 直接纠错**："Sydney's airspace isn't closed. Incorrect reports."（无链接）。
- R007 @10.2min 来源检查："Check your sources: reports airspace NOT closed. Possibly TFR over CBD."（给出替代解释）。
- R015 @151.93min 元级批评："Should not be claiming 'ISIS-style' flag… Can we not propagate unsubstantiated claims yet."
- 同源回声：R002/R004/R008/R014 四条共享同一 URL（cbc.ca/1.2873068）@1.2-78.6min。
- 页面核实：该 CBC 链接当前指向"受害者确认"页面，**内容与 RT 时刻的疏散报道不同**——同源 URL 的内容漂移实证。Wayback 短链无存档。
- 结论：纠错（0.4min）与回声（1.2min 起）几乎同时开始；false 线程上有效纠错存在但被回声淹没——正是候选机制的目标场景。

### 7. SES-P0-028 [false] charliehebdo ★核心配对候选
- R001 @3.65min 同文引述 @BBCBreaking（同源复述）。
- **R003 @11.28min 纠错**：检方否认死亡报告，附 bbc.in/14ulyLt 链接（15min 截点内、带链接）。
- **R004 @14.42min 是 R003 的树内回复**："Correction by prosecutor saying no one killed"——纠错沿对话边传播的直接证据。
- 干扰材料：R006 @22.1min 引入 awd-news 阴谋链接（"Al Qaeda fingerprints"）。
- 页面核实：bbc.in/14ulyLt Wayback 无存档；当前重定向到 BBC live 页（内容不可回溯）→ `TEMPORAL_AVAILABILITY_UNKNOWN`；Guardian live blog（源帖链接）未单独访问。

### 8. SES-P0-029 [false] germanwings-crash
- 阴谋论 thread（"co-pilot was Muslim convert"）；annotation 携带 FOR blog（gatewaypundit）+ **AGAINST blog（gawker antiviral 辟谣）**。
- 树内 360min 无任何纠错；R017 @521.62min 才出现"it's not true sir"（AFTER_CUTOFF）。
- 结论：纠错材料存在于 annotation 层但从未进入截点内的树讨论——"树外纠错不可见"量化案例。

### 9. SES-P0-030 [false] ebola-essien ★核心 false 案例
- 同源回声链：R001-R019 大量逐字引述谣言源头 @YuryAlkaev（5.75-85min）。
- **错误材料在树内传播**：R009 @35.18min 与 R010 @36.43min 传播伪造的"俱乐部发言人"引语（延续谣言），R020 @85.82min 转发该引语——错误材料获得二次传播。
- 来源质询：R016 @78.48min "provide ur source or shut up"。
- annotation 携带 **4 条 AGAINST 链接**（当事人 Instagram 自述 + 3 家媒体辟谣），但树内 360min 无任何行引用它们。
- 页面核实：bleacherreport AGAINST 链接 Wayback 有存档（2014-12-06，距事件约两个月）；Instagram 链接当前返回 200 但为登录墙，无法核内容。
- 结论：正确纠错材料存在（标注层）但从未进入树内；错误引述反而获得传播——依赖感知聚合要解决的核心反差。

### 10. SES-P0-036 [false] charliehebdo
- 同源回声：R002-R019 同文引述 @HuffPostCanada（2.05-232.18min，跨度 4 小时）。
- annotation 富标注：2 FOR（instagram/facebook）+ 1 observing（huffpost）+ **2 AGAINST**（businessinsider"伪造 Banksy 署名"辟谣 + emergent.info 专门谣言追溯条目）+ 1 buzzfeed 否认。
- 树内 360min 无行引用 AGAINST 材料。
- 页面核实：未单独访问（annotation 事实已在包内）。

### 11. SES-P0-053 [unverified] ferguson ★配对候选
- 源帖来自 @TheAnonMessage2（匿名集体账号，先天可靠性低）；R001 @0.0min 完全相同文本。
- 质疑密集：R005 @1.37min（否认+推理）、R011/R012 @7.95-8.28min（照片身份）、R017 @10.62min（"LIES"）。
- **R013 @9.03min 引用当事人先前否认**："didn't the manager of the store already say this was not true"——引用式纠错但无链接。
- annotation：2 条 FOR news-media（usatoday、foxnews）+ 1 条 AGAINST social-media（youtube）。
- 页面核实：usatoday FOR 链接 Wayback 有存档（2014-08-16，事件次日，接近截点）。
- 不确定点：R013 所指"经理否认"是否有当时公开报道可核——留给双人标注阶段。

### 12. SES-P0-056 [unverified] charliehebdo
- R001 @0.5min 同文引述 @SkyNewsBreak（同源复述）。
- 语义区分质询：R015 @16.7min "have they actually been caught or just located"——区分"被捕获"与"被定位"两个命题。
- 不确定纠错：R009 @5.62min "think one already gave himself up earlier mate, could be wrong though"（自我标注不确定）。
- 树深达 17 层，大量离题宗教争论——离题噪声的极端案例。
- 页面核实：无。

## 配对候选汇总（provisional）

| 案例 | V | 同源复述 | 纠错形态 | 截点内 | 材料可核性 |
|---|---|---|---|---|---|
| 026 | false | URL 簇 4 人 | 断言式（0.4min）+ 来源检查 + 元批评 | ✓ | 事实明确；链接内容已漂移 |
| 028 | false | R001 引述 | 引用式（11.28min，带链接）+ 树内传播 | ✓ | 链接 Wayback 无存档，UNKNOWN |
| 014 | true | 引述簇 | 归属质疑→树内确认（6.3-10.8min） | ✓ | 确认引用 CBC TV，无直接链接 |
| 007 | true | 双 URL 簇 4-5 人 | 更正请求（14.4min/28.4min） | ✓ | 短链无存档，UNKNOWN |
| 053 | unverified | R001 同文 | 引用式（9.03min，无链接） | ✓ | 所引"经理否认"待核 |
| 030 | false | 大回声链 | （反向案例：错误引述传播） | ✓ | annotation AGAINST 有存档 |

- 严格满足"同源复述 + 有效独立纠错"的暂定配对：**5 条**（026、028、014、007、053），覆盖 5 个话题，true 2 / false 2 / unverified 1；030 作为反向配对（复述+错误材料传播）补充。
- **缺额说明**：距 P1 准入线（≥6 条）差 1 条强配对；缺的是"树内纠错附带可核历史材料"这一项——028/007 的链接在 Wayback 无存档是主要缺口，属数据可得性限制而非标注不可行。
