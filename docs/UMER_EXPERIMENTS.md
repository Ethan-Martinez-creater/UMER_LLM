# UMER 基本实验情况

> 正式设置(冻结):partition seed 3090、training seed 2000、五折事件级分层、clean DeBERTa-v3-large 初始化、物理/有效 event batch 4/64、max epochs 30、patience 7、类权重幂 0.5、2 随机训练视图、跨视图一致性权重 0.5、图/文本辅助权重 0.15/0.15、线性融合、exact-replay SAM rho=0.05、validation-only `legacy_four_metric_min` 选点。
> 机器可读证据:`results/pheme/authoritative_json/`、`results/maweibo/authoritative_json/`、`results/pheme/ablation_json/`(逐折 result.json、history.json、summary.json 均已复制)。
> 大型资产(数据集、DeBERTa 权重、五折检查点)为只读复现资产,其位置与来源映射记录在内部工作区指南 `WORKSPACE_GUIDE.md`,不随公开仓库分发。

## 1. PHEME 五折正式结果

PHEME(英语 Twitter,9 个新闻话题,6,425 事件:2,402 谣言/4,023 非谣言,105,238 帖,677 个 source-only 事件):

| 折 | Accuracy | Weighted-F1 | Macro-F1 | Rumor-F1 | 阈值 | 入选epoch/总epoch |
|---|---|---|---|---|---|---|
| 1 | 0.9043 | 0.9040 | 0.8972 | 0.8701 | 0.565 | 14 / 21 |
| 2 | 0.8957 | 0.8963 | 0.8901 | 0.8652 | 0.505 | 11 / 18 |
| 3 | 0.8942 | 0.8941 | 0.8868 | 0.8580 | 0.710 | 9 / 16 |
| 4 | 0.8887 | 0.8897 | 0.8833 | 0.8583 | 0.250 | 9 / 16 |
| 5 | 0.9113 | 0.9113 | 0.9053 | 0.8815 | 0.250 | 10 / 17 |
| **均值** | **0.8988** | **0.8991** | **0.8925** | **0.8666** | — | — |
| 标准差 | 0.0080 | 0.0077 | 0.0078 | 0.0087 | — | — |

(指标由每折预测与 confusion matrix 计算;Rumor-F1 为谣言类 F1。)

## 2. Ma-Weibo 五折正式结果

Ma-Weibo(中文微博,4,664 事件:2,313/2,351,1,771,329 帖,6 个 source-only 事件):

| 折 | Accuracy | Weighted-F1 | Macro-F1 | Rumor-F1 | 阈值 |
|---|---|---|---|---|---|
| 1 | 0.9721 | 0.9721 | 0.9721 | 0.9725 | 0.730 |
| 2 | 0.9743 | 0.9743 | 0.9743 | 0.9743 | 0.740 |
| 3 | 0.9721 | 0.9721 | 0.9721 | 0.9726 | 0.525 |
| 4 | 0.9636 | 0.9635 | 0.9635 | 0.9643 | 0.670 |
| 5 | 0.9624 | 0.9624 | 0.9624 | 0.9631 | 0.555 |
| **均值** | **0.9689** | **0.9689** | **0.9689** | **0.9694** | — |
| 标准差 | 0.0049 | 0.0049 | 0.0049 | 0.0047 | — |

(Ma-Weibo 宏平均 P/R/F1 分别为 0.9698/0.9691/0.9689;官方记录见 summary.json。)

## 3. PHEME 组件消融(五折,同种子)

移除单一证据源(其余设置不变,`--ablate-evidence`), Accuracy 相对完整 UMER 的下降:

| 变体 | Accuracy | Weighted-F1 | Macro-F1 | Rumor-F1 | ΔAccuracy |
|---|---|---|---|---|---|
| 完整 UMER | 0.8988 | 0.8991 | 0.8925 | 0.8666 | — |
| 去文本(no_text) | 0.8601 | 0.8609 | 0.8525 | 0.8193 | **-3.88 pt** |
| 去一致性(no_consistency) | 0.8895 | 0.8902 | 0.8837 | 0.8579 | -0.93 pt |
| 去图(no_graph) | 0.8940 | 0.8942 | 0.8873 | 0.8601 | -0.48 pt |
| 去检索记忆(no_memory) | 0.8985 | 0.8988 | 0.8923 | 0.8663 | -0.03 pt |

结论:文本证据贡献最大,跨视图一致性次之,图证据第三,SAM 增益约 +0.37 pt(相对 Round21 非 SAM 版),跨事件检索记忆聚合增益最小(+0.03 pt)但仍作为泄漏安全的证据源保留——不得在没有预定义子群研究的情况下夸大其作用。

## 4. 训练协议关键事实

- 每折学习率组:graph 1e-4 / DeBERTa 2e-5 / fusion 1e-3;AdamW wd=0.01;线性 10% warmup;
- 类权重(PHEME fold1):[1.0, 1.2944](幂 0.5 逆频率);
- SAM 记录(run result.json `sam` 字段):rho=0.05,扰动范围=全部可训练参数,两遍同 effective batch 64,RNG 重放=True,视图重放=True,梯度裁剪仅第二遍;
- 每折 result.json 记录完整复现元数据(阈值选择范围=当前内层验证、epoch 监控范围=held-out 内层验证、检索记忆范围=严格训练折 leave-one-out 等);
- 五折脚本:`scripts/run_pheme_fivefold.sh <exp_id>` / `run_maweibo_fivefold.sh`,折完自动 `summarize_round_018_seed.py` 生成 summary.json;detached 版本为 `launch_*.sh`;
- 研究筛选规则:任何新模型级提议先跑 fold1+fold2,报告双折与均值,预注册判据通过且无实质退化才准跑 fold3–5;
- PHEME 与 Ma-Weibo 不得在同一 GPU 上并发正式运行。

## 5. 复现资产

数据集(PHEME 处理图 605 MB / 原始线程 1.1 GB;Ma-Weibo 处理图 16 GB / 原始 3.8 GB)、DeBERTa-v3-large 权重(836 MB)与两个数据集的五折检查点(各 8.4 GB)为只读复现资产,规模合计约 38 GB,不含在本仓库中;其部署位置、来源映射与运行环境记录在内部指南 `WORKSPACE_GUIDE.md`(不随公开仓库分发)。仓库内的 `results/` 只保存机器可读的 JSON 证据。

## 6. 历史优化轨迹(从何而来)

UMER 是原工作区 32 轮单变量增量优化的收敛点,关键里程碑:

- round_006:节点级 DeBERTa 文本特征(384 维节点嵌入来源);
- round_009:四标准视图 + 多视图集成;
- round_013:图+文本+检索三路联合训练(`JointTriFusionModel`,本模型主体);
- round_021:跨视图广义 JSD 一致性(0.5);
- round_025:分组学习率与 layerwise decay 机制(正式用 1.0,即不衰减);
- round_032:effective-batch 级 exact-replay SAM(rho=0.05),即最终配置。

早期轮次(001–005、011、015、026、031 等)的完整代码不在本仓库(仅保留 UMER 依赖闭包);本仓库 `project/optimization_rounds/` 内各保留轮的 README 记录了对应的单变量设计。
