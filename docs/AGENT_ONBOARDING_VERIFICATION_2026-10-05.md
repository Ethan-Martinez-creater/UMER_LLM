# 新执行智能体接手验证报告

日期：2026-10-05。性质：接手验证记录（`UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md` §36 接手流程的执行结果），不构成新实验、新协议或阶段授权。

## 1. 版本一致性

| 位置 | HEAD | 状态 |
|---|---|---|
| LOCAL | `d71991dd9745a910e94ca91398d0b871a7ffda7f` | 与 origin/main 一致（fetch 经代理确认，无 ahead/behind） |
| GitHub origin/main | `d71991dd9745a910e94ca91398d0b871a7ffda7f` | 最新 commit 2026-10-03（M1-F COMPLETED） |
| SERVER `/data/jyz/next/llm/cr_tser_ws` | `6870220a93c9d450e8c65f8bf5fd2b87b0acc628` | 落后 LOCAL 一个 commit；该 commit（d71991d）是 LOCAL 收集 SERVER M1-F 产物后的提交，属预期状态，SERVER 无需回推。SERVER 未跟踪变更 = M1-F 产物（已在 d71991d 提交）+ 历史 `utility_labels` 目录，无漂移 |

## 2. 本机验证结果（conda env `pytorch`，Python 3.9.18）

```text
python -m pytest project/bcr_utility/tests -q   => 277 passed (493.83s)
python -m pytest project/cr_tser/tests -q       => 219 passed (37.37s)
python -m compileall project/bcr_utility scripts/bcr_*.py => exit 0
```

## 3. 服务器验证结果

```text
连接入口      = next/.codex_tmp_ssh_run.cmd（隧道正常，单次连通）
DGPA          = Python 3.11.13 / torch 2.7.1+cu128 / CUDA available
SERVER verdict 参考 = results/bcr_utility_v1/verifier/bcr_verify_m1f_server.json
                  mode=m1f, issues=[], pending=[]（未重跑，读取既有正式产物）
```

## 4. 环境事实修正（补充交接手册 §3/§39）

- 本机 conda `pytorch` 环境实际位于 `E:\anaconda\envs\pytorch`（miniconda3 base 里没有 pytest，直接 `conda activate pytorch` 在 miniconda 下会失败）；
- `next/` 连接目录实际路径为 `E:\Graduate_work_folder\Graduate_Project_Worksapace\Algorithm\modified\next\`（含 `.codex_tmp_ssh_run.cmd` / `.codex_tmp_scp_run.cmd`，ssh `-p 32985 jyz@60.215.128.50`）；
- GitHub 直连不稳定（`github.com:443` 反复连接超时），需要时对 git 使用命令级代理参数 `git -c http.proxy=http://127.0.0.1:7890 fetch/push`，不修改全局 git 配置。

## 5. 三条研究线的当前状态（以远端 M1-F 与最新评审为准）

1. **CR-TSER**：CLOSED / NO-GO（历史冻结，只读）。
2. **TC-DSCR**：正式证据已闭合（`results/tcdscr/final_evidence/FINAL_EVIDENCE_REPORT.md`）——Gap A static held-out 两侧 STRONG_SUPPORT；Gap F 确认 DYNAMIC_V1_NOT_SUPPORTED；Gap B+C 最终 reader 证据：PHEME MS vs Static +0.0291（CI 跨 0，WEAK_PASS），Ma-Weibo −0.0130（FAIL）。最终定位 `MS_TSR_COMPRESSION_ONLY`，MS-TSR 只能作为压缩组件主张，负结果台账 N01–N04 不可软化。
3. **BCR-Utility**：M1（ZERO-TOUCH FAIL / LIGHT-TOUCH PASS → M1_CONDITIONAL_GO）→ M1-E（fingerprint 无正增量，D5 相对 B0 +0.0710）→ **M1-F：D5 相对 dev 选出的强基线 S2_source_state 平均 Macro-F1 −0.0075（CI 跨 0），active Macro-F1 −0.1893，收益仅 NEUTRAL → `M1F_CLOSE_BCR_UTILITY_METHOD`**。整个 bcr_v1 方法主线关闭；M2 未进入、未授权。

## 6. 研究方向现状（等待用户决策）

- `docs/research_review_2026-10-02/`：10-02 文献评审 + 只读算术审计（D5 相对常数基线仅 +0.0086）；
- `docs/check/LLM_RUMOR_RESEARCH_DECISION_2026-10-03.md`：BCR 关闭决策 + A0 准入审计计划（R2 记录用户已暂不考虑任务语义错位审计方向）；
- `docs/check/LLM_SOCIAL_MISINFORMATION_DIRECTION_SEARCH_R2_2026-10-04.md`：第二轮选题检索（条件推荐：数值/时间主张范围核验 TSVer；社会对话：局部立场可组合边界 TruthStance；多模态较弱）；
- `docs/check/LLM_SOCIAL_RUMOR_THESIS_DIRECTION_SEARCH_R3_2026-10-04.md`：第三轮完整任务评估（候选 A 受操纵社会上下文下的安全研判认证 = 需用户确认是否接受"安全属性研究"形态；候选 B 协调放大发现 = 数据阻碍大；候选 C 共享核查与预算调度 = 条件备选）。
- **当前没有任何候选立项；第二论文方向等待用户选择。**

## 7. 本次提交内容说明

- 新增本报告；
- 一并提交工作区遗留的未跟踪文档：交接手册 `UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md`、`docs/research_review_2026-10-02/`、`docs/check/`、`docs/old_plan_md/` 内两个新文档，以及既有 docs 文档移入 `docs/old_plan_md/` 的整理变更（22 个移动，无内容修改）；
- 未提交：`.tmp_v3_launch/`、`.bcr_check_*/` 临时目录，`results/tcdscr/` 下未跟踪的 `.pt` 检查点与 `diagnosis.log`（服务器产物，不入库）。

接手验证完成，遵守 fail-closed 与"执行一轮 → commit → push → STOP → 研究审批"纪律，等待用户对下一步研究方向的决策。
