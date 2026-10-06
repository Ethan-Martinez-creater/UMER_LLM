# SES-v1 P0 总结报告

- 日期：2026-10-06。总判决：**`SES_P0_NOT_READY`**（三项缺口，见下）；无模型实验、无 GPU、无训练、无新 gold。
- 完整证据：[ASSET_AUDIT](P0_ASSET_AUDIT.json) | [MANIFEST](P0_MANIFEST.csv) | [OBSERVABILITY](P0_OBSERVABILITY.csv) | [CASE_AUDIT](P0_CASE_AUDIT.md) | [NEAREST_WORKS](P0_NEAREST_WORKS.md) | [VERDICT](P0_VERDICT.json) | [DRAFT](P0_NEXT_STAGE_DRAFT.md) | [VALIDATION](P0_VALIDATION.json)

## 覆盖分母与关键数量

- 数据：figshare PHEME veracity v1（CC BY 4.0，46,529,729 字节，SHA256=079f6ffd…）安全解包；6,425 threads，2,402 rumour 全部有原生 V（true 1,067 / false 638 / unverified 697，**冲突 0、缺失 0**）。
- 固定盐 `SES-v1-P0-20261006` 选样 **60 条（各类 20，无缺额，9 个话题全覆盖）**，重放一致（P0_VALIDATION.json：选样重放、唯一性、R/V 语义、数值列一致性、无原始 ID 泄漏全部通过；单元测试 6/6）。
- 60 条全部完成 6 维 provisional 审查（15 条 15min 无候选）；12 条深查（树深最大 17 层）；公开页面请求 9/12。

## 核心发现

1. **依赖结构可程序化恢复**：RT/引述前缀、共享 URL 簇（11/60 条）、完整父子树、PHEME 原生 annotation 链接自带 **for/against/observing** 立场标注（此前项目未使用）。
2. **五条强配对候选**（同源复述+有效独立纠错）：026（0.4min 纠错 + 4 人 URL 簇）、028（11.3min 检方否认 + 纠错沿树传播）、014（归属质疑→树内确认 6.3-10.8min）、007（双 BBC URL 簇 + 更正请求）、053（引用经理否认）；030 为反向案例（错误引述传播、annotation 4 条 AGAINST 从未进入树内）。
3. **历史可用性缺口**：候选纠错材料的短链在 Wayback 无存档且当前内容漂移（cbc.ca 链接现为完全不同的报道）——"当前可读 ≠ 截点时内容"获得实证；仅 usatoday（事件次日快照）等少数长链可核。

## 近邻结论

9 篇中 7 篇已核（4 正文 / 2 摘要 / 1 元数据受限），**无一篇同时覆盖真实线程依赖结构 + 同源复述不重复计数 + 独立纠错保留**；但 Whose Facts Win（ACL 2026）已在合成条件实证同源重复被当作事实支持、Speaker-Free Floor 已区分重复条数与独立说话人数——**现象级主张已被占位，贡献只能落在真实线程的结构化机制与纠错保留**。LLM-TKT 全文未获取、GAVEL 仅摘要 → `nearest_work_exclusion=UNRESOLVED`。

## 总判决与缺口

`SES_P0_NOT_READY`：条件 3 差 1 条强配对（缺"树内纠错附带可核历史材料"）、条件 4 历史可用性大多 UNKNOWN、条件 5 两篇近邻全文未解。前两项部分现实可补（定向存档核查 + P1 增加深查），全文项可在 P1 评审前再试作者版。两条可证伪机制假设（H1 依赖感知计数、H2 可溯源纠错保留）已就绪，含最强基线与推翻条件。

下一轮：见 [P0_NEXT_STAGE_DRAFT.md](P0_NEXT_STAGE_DRAFT.md)（人员 PENDING_HUMAN_RESOURCE）；等待研究评审，未进入 P1。
