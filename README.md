# UMER:联合图–文本–跨事件检索的单检查点谣言检测器

UMER(**U**nified **M**ulti-evidence **E**vent **R**umor detector)是一个单检查点事件级社交媒体谣言检测模型,输入为一个传播事件(源帖+240 小时窗口内的回复树),输出 `rumor / non-rumor` 二分类预测。它将三类互补证据联合优化进同一个检查点:

- **图证据**:置换不变的节点 Set 编码器 + 一层节点 token Transformer;
- **文本证据**:DeBERTa-v3-large 对同一事件的多个确定性序列化视图;
- **跨事件语义证据**:从当前训练分区记忆库检索的近邻事件 rumor log-odds。

推理时是一个普通前向网络——无 LLM、无外部检索、无模型集成。历史内部代号 `round_032_sam` 仅为可追溯性保留,方法名统一为 UMER。

本仓库同时完整记录了 2026 年围绕 UMER 展开的**全部 LLM 结合尝试及其失败结论**(见下文),作为后续研究的负证据与设计约束。

---

## 1. 模型组成

详细架构(精确到模块、维度、损失与源码位置)见 [`docs/UMER_MODEL_ARCHITECTURE.md`](docs/UMER_MODEL_ARCHITECTURE.md)。要点:

### 1.1 输入

每个事件为以源帖为根的对话树:

- `node_feat` `[max_nodes, 395]`:384 维节点语义嵌入 + 11 维节点/用户特征;
- `struct_feat` `[max_nodes, 1024]`:邻接签名 + 3 维稠密传播摘要(度、深度、相对时间);
- 文本:源帖恒在首位、回复按 4 种确定性方式序列化(full / reversed / even / odd),以 ` [SEP] ` 连接,DeBERTa tokenizer,`max_length=256`。

### 1.2 图分支

两个并联读出,事件特征各 768 维:

1. **NodeTokenSetEncoder**(置换不变):14 维节点 token(11 非文本特征 + 3 传播特征)→ 2 层 MLP(256 隐层)→ mean/max 池化拼接(512)→ 投影到 768;
2. **OriginalGraphBranch**:节点投影(395→768)与结构投影(邻接签名 1021→256、传播摘要 3→256 拼接后→768)相加,加一个可学习 CLS token,过一层 TransformerEncoderLayer(d_model=768, 8 头, FFN 1536, GELU, pre-norm),读出 CLS ⊕ 均值池化 → 768。

两读出拼接(1536)→ 融合投影 → 分类头输出 **2 维图 logits**。图分支参数量 8,526,850(代码硬校验)。

### 1.3 文本分支

`DeBERTa-v3-large` 序列分类器(num_labels=2)。训练时每事件随机抽 2 个不同视图(均受监督);验证/测试用全部 4 视图取 logits 均值。

### 1.4 跨事件检索分支

事件 384 维文本嵌入的 4 个统计块(源帖、均值、最大值、均值-源差,各归一化拼接)构成 1536 维查询;与**当前外层训练折**事件记忆库做余弦检索 top-k=31,softmax 温度 0.05 加权近邻标签得 rumor 概率,取 log-odds 输出 **1 维检索 logit**。训练查询 leave-one-out;验证/测试事件从不写入记忆库(严格防泄漏)。

### 1.5 融合与损失

线性融合头 `Linear(5→2)`(输入 = 图 logits 2 + 文本 logits 2 + 检索 logit 1)。总损失:

```text
L = CE(最终logits, y)
  + 0.15 · CE(图logits, y)                    # 图分支辅助
  + 0.15 · CE(各受监督文本视图logits, y)        # 文本分支辅助
  + 0.5  · 广义JSD(两视图预测分布)             # 跨视图一致性
```

类别权重为平方根逆频率(`(n_0/n_1)^0.5`)。

### 1.6 训练:exact-replay SAM

- AdamW(wd=0.01),分组学习率:图 1e-4 / DeBERTa 2e-5 / 融合 1e-3;线性 10% warmup + 衰减;
- 物理 event batch 4 × 梯度累积 16 = **有效 event batch 64**;AMP;
- **exact-replay SAM(rho=0.05)**:SAM 要求两遍同 batch,与梯度累积冲突;实现按有效 batch 窗口执行——第一遍累计梯度后以全局范数构造扰动,第二遍**重放同一 16 个微批次并逐个恢复 CPU/CUDA RNG 状态**,使 dropout 掩码与视图采样在两遍间完全一致;仅第二遍梯度被裁剪并送入 AdamW。训练计算量约 2×,推理零额外成本。

### 1.7 划分与选择

五折事件级分层(partition seed 3090),每折内层 10% 标签分层的验证池(training seed 2000);validation-only `legacy_four_metric_min` 选 checkpoint 与阈值;held-out 外层测试折仅在训练完成后评估一次;交付物为每折一个 checkpoint,无 checkpoint 平均、无测试时融合。

## 2. 基本实验情况

完整逐折数值见 [`docs/UMER_EXPERIMENTS.md`](docs/UMER_EXPERIMENTS.md);机器可读证据在 `results/`。

### 2.1 PHEME(英语 Twitter,6,425 事件:2,402 谣言 / 4,023 非谣言)

| 指标 | 五折均值 | 标准差 |
|---|---|---|
| Accuracy | **0.8988** | 0.0080 |
| Weighted-F1 | **0.8991** | 0.0077 |
| Macro-F1 | **0.8925** | 0.0078 |
| Rumor-F1 | **0.8666** | 0.0087 |

### 2.2 Ma-Weibo(中文微博,4,664 事件:2,313 / 2,351)

| 指标 | 五折均值 | 标准差 |
|---|---|---|
| Accuracy | **0.9689** | 0.0049 |
| Weighted-F1 | **0.9689** | 0.0049 |
| Macro-F1 | **0.9689** | 0.0049 |
| Rumor-F1 | **0.9694** | 0.0047 |

(宏平均 Precision/Recall/F1 = 0.9698 / 0.9691 / 0.9689。)

### 2.3 PHEME 组件消融(五折,单变量移除)

| 变体 | Accuracy | Δ |
|---|---|---|
| 完整 UMER | 0.8988 | — |
| 去文本 | 0.8601 | **-3.88 pt** |
| 去跨视图一致性 | 0.8895 | -0.93 pt |
| 去图 | 0.8940 | -0.48 pt |
| 去跨事件检索记忆 | 0.8985 | -0.03 pt |

文本证据贡献最大;跨事件检索分支聚合增益小,保留为泄漏安全的证据源但不应夸大。SAM 相对无 SAM 版本增益约 +0.37 pt。

### 2.4 复现要点

冻结配置:partition seed 3090 / training seed 2000 / batch 4×16 / max epochs 30 / patience 7 / 类权重幂 0.5 / 一致性权重 0.5 / 辅助权重 0.15+0.15 / 线性融合 / SAM rho 0.05。任何新模型级提议须一次只改一个变量,先跑 fold1+fold2 双折筛查,预注册判据通过且无实质退化才继续 fold3–5。数据集、DeBERTa 权重与五折检查点(合计约 38 GB)不含在仓库内;`results/` 保存逐折 result/history/summary JSON。

## 3. 已尝试的 LLM 结合方案(全部失败,负证据存档)

2026-08-09 至 08-15,围绕"如何将 LLM 与 UMER 结合"进行了四个阶段、30+ 个预注册实验,使用的模型为本地部署的 Qwen3-8B(主)与 Qwen3-0.6B(对照)。**除一处数据质量审计通过外,全部判 FAIL 或 NOT_READY。** 逐项设计、数值与失败原因见 [`docs/LLM_INTEGRATION_ATTEMPTS.md`](docs/LLM_INTEGRATION_ATTEMPTS.md);此处为索引:

### 3.1 LLM 认知蒸馏与审计串行链(Round033–054,22 轮)

以 8B teacher 对全部 6,425 PHEME 事件的 label-blind 结构化认知标注(stance/role/salience 等 18 维)为原料:

- **事件级认知蒸馏**(033)、**认知视图蒸馏**(034)、**节点级 stance/role/salience 蒸馏**(039):LLM 输出作训练期辅助监督,推理时不出现——双折门槛均未过;
- **风格鲁棒改写+审计**(036/040):仅 66.67% 语义对通过双向蕴含审计;
- **虚拟回复生成**(037)与**合成反应干预**(041):grounding 严格可用率仅 27%,禁入训练;
- **认知选择性路由**(043)、**salience 干预**(044)、**反事实解释一致性**(045)、**因果 salience 曲线**(047):各保真门槛失败;
- **crossfit 小 teacher**(046):LoRA 蒸馏 0.6B 达到 schema 有效率 0.988、峰值显存仅 8B 的 8.9%,但耗时比与下游非劣性门槛失败——全链最接近可行的一轮;
- **可抽取证据卡**(048)/ **选择性解释**(049):认知覆盖显著优于对照但保真度未可靠超过时间/哈希基线;
- **teacher 缺失审计**(050):链上唯一全过(可用覆盖率 99.91%),但属数据质量工作;
- **跨事件主张谱系**(051)/ **纠正时延监测**(052)/ **活动简报**(053)/ **干预机会回放**(054):跨事件关系严格有效率仅 49.25%;8B 对打乱关系 51.92% 接受率;干预回放日期稳定性 p=0.125、CI 跨零。

### 3.2 跨事件纠正关系人工审核集(v1/v2/v3)

- v1(400 条,双盲+裁决):九分类 κ=0.1346,裁决后仅 7 条有效纠正;
- v2(协议校准后全新 400 条):纠正 vs 其他 κ=0.8047 一度过线,但在独立富集集上**复现失败**(κ=0.4236,共同阳性 8);
- v3(三阶段机械派生协议):96 条 sealed pilot 因标注员理解测验 0/16 未达 16/16 按协议停机,pilot 从未开放;合成基准上 0.6B/8B 合法证书均为 0/16。

### 3.3 跨领域调研后六方向(A–F)

基于九类 LLM 赋能范式的文献调研,预注册实验:目标驱动发现(A1:合法候选 0)、多粒度事件理解(C1:可用记录 1/60,staged 分解使引用合法率 .35→.767 为工程信号)、开放词汇策略(B1:codebook 混入真值判断,heldout 未开放)、时间截断 KG(D1:无外部档案索引,NOT READY)、人机分歧机制地图(E1:双来源证书完整率 0)、可控仿真(F1:无真实行为校准,NOT READY)。结论:无一方向同时满足创新性与可行性门槛。

### 3.4 时限开放世界谣言核查(TOWRV)

重新定义任务(历史截点上的四状态证据判断与拒答),完成协议冻结、候选 manifest、20 事件原子主张分解三审修正;终止于外部证据可得性审计:档案类站点全部不可达,严格时间依据下合格事件预估 <20%,判 NOT_READY。

### 3.5 横向结论(为何全部失败)

1. **接口失败**:8B 在受约束 JSON/证书/引用任务上 schema 违规率极高;
2. **职责未解耦**:让 LLM 同时做语义判断与结构化输出/证书绑定时必然污染;唯一通过的数据审计与最接近可行的小 teacher 均把 LLM 限制为纯语义角色;
3. **人工一致性天花板**:纠正关系双盲 κ 始终在 0.13–0.43,gold 无法建立;
4. **外部资源缺失**:时间约束事实核查需要带可靠首次可得时间的档案;
5. **合成缺校准**:虚拟行为与干预效果缺真实数据锚定。

由此沉淀的五条设计约束:LLM 只承担语义角色(ID/文本/哈希/证书由程序绑定);不以"LLM 提升分类精度"为主要贡献;人工 gold 路线先做一致性 pilot(κ<0.7 即停);外部证据路线先做可得性审计(NOT_READY ≠ FAIL);保持冻结协议、预注册门槛与 fail-closed 判决,失败不得事后挽救。

## 4. 仓库结构

```text
UMER/
  README.md                     本文件
  docs/
    UMER_MODEL_ARCHITECTURE.md  模型组成(模块/维度/损失/SAM)
    UMER_EXPERIMENTS.md         五折正式结果、消融、训练协议
    LLM_INTEGRATION_ATTEMPTS.md 已尝试 LLM 结合方案全记录
  project/                      UMER 最小可运行闭包
    optimization_rounds/        006/009/013/016/021/022/024/025/032
                                (+033/034/039 仅因 import 依赖保留,正式不启用)
    src/rumor_detection/        数据/模型/训练/工具
    protocol/ runner/ analysis/ config/ experiments/ ablation_studies/
  scripts/                      入口:工作区校验 / 冒烟 / 五折运行与汇总
  results/                      PHEME 与 Ma-Weibo 逐折 JSON 证据 + 消融
  resources/CORE_REFERENCES.md  数据集与核心方法文献
  requirements.lock.txt         依赖锁定
```

## 5. 引用与数据

PHEME:Zubiaga et al., *Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads*, PLOS ONE 2016。Ma-Weibo:Ma et al., *Detect Rumors from Microblogs with Recurrent Neural Networks*, IJCAI 2016。其余核心文献见 [`resources/CORE_REFERENCES.md`](resources/CORE_REFERENCES.md)。

数据集与预训练权重依其原始许可使用,不含在本仓库中。
