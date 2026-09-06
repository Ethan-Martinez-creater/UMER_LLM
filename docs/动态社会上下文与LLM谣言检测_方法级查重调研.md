# 第一候选方向方法级查重调研：动态社会上下文与 LLM 谣言检测

## 1. 本轮调研结论

这一轮把第一候选拆到“方法级”以后，结论比上一轮更精确了：

**“动态/早期 + LLM”本身已经不能算研究空白，“动态图 + LLM”这个说法也过宽。真正还比较干净的缺口，是如何在传播图不断演化的过程中，为 LLM 动态构造一个紧凑、结构化、严格时间因果的 social context。**

也就是说，如果继续这条线，研究问题需要从：

> 动态谣言检测 + LLM

收窄为：

> **Dynamic Social-Context Refinement for LLM-based Rumor Detection**  
> 即：面对不断增长的传播树，当前时刻究竟应该把哪些社会传播证据、以什么结构提供给 LLM？

---

## 2. NAACL 2025 SePro：已解决“静态大规模 social context 怎么给 LLM”，但没有解决动态问题

SePro 是目前与你这个候选最接近、也最需要认真规避的工作。

它首先发现两个问题：

1. LLM 面对大量评论容易 “lost in the middle”；
2. LLM 不擅长直接理解 propagation graph 这种结构化数据。

因此它先建立 propagation graph 和 semantic graph，用 GAT 建模，然后根据 attention 选取关键节点，把原来很大的 social context 重构为适合 LLM 的中等规模上下文，最后让 LLM 根据 claim、local context、global context 和小模型预测分布进行验证和解释。

其基本流程可以概括为：

\[
G
\rightarrow GAT
\rightarrow Important\ Nodes
\rightarrow Compact\ Social\ Context
\rightarrow LLM
\]

所以如果后续只是让 UMER：

\[
G\rightarrow TopK\ replies\rightarrow LLM
\]

那和 SePro 太接近。

但关键区别是，SePro 的问题定义仍然是一个完整样本：

\[
x_i=\{claim,c_1,\ldots,c_n,P\}
\]

即已经获得这个事件的评论和传播结构以后进行检测，并没有定义：

\[
G_{t_1}\subset G_{t_2}\subset\cdots
\]

这种严格在线的传播过程。

更重要的是，SePro 自己在 Limitation 中承认：

- 所谓“moderate-size context”没有理论依据；
- 最终主要靠超参数实验确定，例如约 50 个 global comments + 50 个 local comments；
- 不同 foundation model 泛化表现不稳定，特别是在 LLaMA3 系列上表现较差；
- Twitter 等旧数据可能已经进入新 LLM 的训练数据，存在 benchmark contamination 风险。

因此 SePro 留下的一个真实问题是：

\[
\boxed{
\text{什么是“足够且必要”的 social context？}
}
\]

而它自己解决的只是**静态条件下的经验式 context refinement**。

---

## 3. KDD 2026 Few-Shot EARD：已解决“什么时候停止等待”，但基本没有利用传播树结构

这是第二篇必须规避的工作。

它把 Early Rumor Detection 建模为 MDP：

\[
s_t\rightarrow
\begin{cases}
Continue\\
Stop
\end{cases}
\]

Agent 学习什么时候停止观察，一旦停止，LLM 做最终 rumor/non-rumor 判断。

因此它已经相当明确地占据了：

\[
\boxed{\text{When should an LLM make an early prediction?}}
\]

这个问题。

而且它覆盖范围比表面上更大：除了 PHEME、Twitter、BEARD，还使用 Twitter-COVID-19，并做了 pre-COVID → COVID 的新事件跨数据集实验。

作者还提出，面对 incomplete observations 和新事件 uncertainty，未来可以把 MDP 扩展成 POMDP。

所以：

> **“Early + Emerging Event + LLM”本身已经不能算一个很干净的研究空白。**

但它有一个非常明显的结构缺口。

其问题定义主要是按时间排序的帖子：

\[
M=\{(m_0,t_0),...,(m_i,t_i)\}
\]

官方代码的数据格式核心只有：

```text
timeline
tids
texts
```

没有 parent-child edge、tree depth、branch structure 等完整传播图关系。

因此它主要解决：

\[
\text{chronological text stream}
\]

而不是：

\[
\text{evolving structured propagation graph}
\]

这为 UMER 留下了空间。

---

## 4. K-FLARE 2026：占据了“传播还没形成时怎么办”

K-FLARE 研究的是一个更极端的早期情况：

\[
Replies\approx0
\]

论文认为，现有方法大量依赖用户回复，但 early stage 时回复可能稀疏、不可靠甚至完全不存在。

因此它设计双流：

\[
\text{Inductive Stream}
\]

使用 semantics、knowledge、topology、temporality 等；

以及：

\[
\text{Deductive Stream}
\]

使用 Language Model 直接判断 source post authenticity。

最后通过 evidence-balancing mechanism 融合两者。

因此这一块已经有人占据：

\[
\boxed{\text{Source-only / Reply-free + LLM}}
\]

但 K-FLARE 研究的是**没有社会上下文的时候如何补偿**。

它没有研究：

\[
G_0
\rightarrow G_{1h}
\rightarrow G_{3h}
\rightarrow G_{6h}
\]

之后随着真实 replies 不断到来：

> LLM 应该如何持续改变自己使用的 social evidence？

因此二者是互补的：

```text
K-FLARE：
没有 social context → 怎么检测？

我们关注的问题：
social context 开始出现并不断增长 → 怎么使用？
```

---

## 5. DCAGTN 2026：动态图侧已经相当成熟，不能再把“加时间信息”当创新

DCAGTN 明确指出已有方法的问题：

- 静态图忽略传播演化；
- 每次重算整图成本高；
- 部分动态图没有严格保持 chronology；
- 会发生 temporal leakage；
- 早期预测弱；
- streaming content 带来实时计算压力。

它因此构造：

\[
\text{Sequential Snapshot}
+
\text{Temporal Snapshot}
\]

然后采用 causal attention 保证信息只沿时间正向传播，再用 lightweight feedback 对新增节点增量更新，不重新计算整个图。

这意味着，如果后续论文只是：

> “把 UMER 改成动态图，加入时间 attention。”

已经不够。

COLING 2025 也已经通过 edge time interval + structural entropy 对 propagation tree 进行 temporal optimization；2025–2026 还有多种 structural-temporal 模型。

因此动态传播这一侧的正确使用方式应该是：

> **作为新研究问题的基础，而不是创新点本身。**

---

## 6. DAWN / WSDM 2025：实验协议也不能继续沿用普通随机划分

DAWN 对后续实验设计非常重要。

它指出传统 social-graph fake news detection 往往先随机切 train/test，再利用完整 social context 构图。

这等于训练时能够看到未来才会出现的用户、评论等信息。

它重新定义 temporality-aware evaluation：

\[
Train=
\text{only information available before }T
\]

\[
Test=
\text{future information after }T
\]

在这种更真实的条件下，多个传统 social-graph 方法的 F1 明显下降。

因此如果后面做动态 LLM–UMER，不能只是：

> 仍然随机五折 → 把每个 test event 自己切成 1h/3h/6h。

这样只能保证**事件内部 temporal causality**，却不能保证**训练集到测试集的 global causality**。

真正严格的实验至少应该同时满足：

\[
\boxed{
\text{Event-internal causality}
+
\text{Train-test causality}
}
\]

这一点应成为后续工作的实验设计基础，而不是模型创新。

---

## 7. 已有“动态图 + LLM”工作，但 LLM 只是解释器

2025 年已有工作使用：

\[
Dynamic\ Graph
\rightarrow GCN
\rightarrow GRU
\rightarrow Classification
\]

然后把：

- detection result；
- rumor text；
- propagation information；

送给 Llama-3-8B，让 LLM 根据按时间排列的 propagation chain 生成解释。

因此以后不能声称：

> “首次结合 dynamic propagation graph 和 LLM。”

这个说法已经不成立。

但该工作的 LLM：

\[
\boxed{\text{只负责解释已有 detector 的输出}}
\]

LLM 并不利用动态图 evidence 参与真实性判断。

因此更准确的空白仍然是：

\[
\boxed{
\text{LLM as a detector/reasoner over evolving structured social evidence}
}
\]

而不是 generic “dynamic graph + LLM”。

---

## 8. 六类核心工作对比

| 工作 | 时间问题 | 使用传播树结构 | LLM作用 | Context选择 | 严格 temporal protocol |
|---|---|---:|---|---|---:|
| **SePro / NAACL25** | 静态 | ✓ | 最终检测+解释 | ✓，GAT选关键节点 | ✗ |
| **Few-Shot EARD / KDD26** | **Early/动态** | 基本✗，时间帖子序列 | **最终检测器** | 主要按时间截断 | 事件内部✓ |
| **K-FLARE / KBS26** | 极早期/reply-free | 部分拓扑/时间 | source内容推理 | 不解决动态context | — |
| **DCAGTN / KBS26** | **实时动态图** | **✓** | ✗ | 图 attention | **✓** |
| **DAWN / WSDM25** | **train-test chronology** | 社会图✓ | ✗ | — | **✓✓** |
| **Dynamic+LLM Explanation / 2025** | 动态图 | ✓ | **只做解释** | propagation chain | 未作为核心问题 |

从这个对照可以看到，几乎所有单独的格子都有人做了。

但是下面这个组合仍然比较空：

\[
\boxed{
\text{Strict Temporal Evolution}
+
\text{Propagation Structure}
+
\text{Adaptive Context Refinement}
+
\text{LLM Reasoning}
}
\]

---

## 9. 第一候选应进一步收窄

第一候选不应该再泛称：

> 动态 LLM–UMER。

更准确的研究问题应该是：

> **在谣言传播过程中，只允许使用当前时刻已经出现的信息，如何从不断演化的传播图中动态选择和组织最有价值的社会证据，使 LLM 能够有效、稳定地完成真实性推理？**

形式上：

\[
G_t=(V_t,E_t),\qquad G_t\subseteq G_{t+\Delta t}
\]

给定一个 LLM context budget \(B\)，不允许简单：

\[
Prompt_t=全部V_t
\]

而是学习：

\[
C_t=S(G_t,X_t,B)
\]

满足：

\[
C_t\subseteq G_t
\]

且绝不允许：

\[
C_t\cap(G_T-G_t)\neq\varnothing
\]

然后：

\[
LLM(Claim,C_t)\rightarrow Verdict
\]

真正研究的是：

\[
S(\cdot)
\]

即：

> **dynamic evidence selection / social-context refinement**

---

## 10. 三个可能创新层的重新判断

### A. 动态 social-context selection —— 最值得继续

核心流程：

\[
G_t
\rightarrow
\text{select structurally + semantically informative evidence}
\rightarrow LLM
\]

SePro 是：

\[
G_{complete}\rightarrow Top\text{-}n
\]

我们要研究的是：

\[
G_1,G_2,\ldots,G_t
\rightarrow
C_1,C_2,\ldots,C_t
\]

而且选出的上下文应该随着传播状态变化。

例如：

早期：

\[
Source+direct\ skeptical\ replies
\]

中期：

\[
Source+representative\ branches
\]

后期：

\[
high\text{-}value\ semantic/structural\ subgraph
\]

不是始终固定“50条评论”。

这一点正好对应 SePro 自己承认的：

> moderate size 缺乏理论依据。

---

### B. “如何把图转换给 LLM”——不适合单独作为创新

SePro 已经把 hierarchical social context 序列化给 LLM；已有动态图+解释工作也已经把 propagation chain 按时间送给 LLM。

因此：

> tree → text serialization

单独做没有足够空间。

除非它服务于 A，例如：

\[
\text{selected dynamic subgraph}
\rightarrow
\text{compact structured representation}
\]

否则不建议作为主要贡献。

---

### C. Continue/Stop / POMDP —— 有价值，但不能作为主创新

KDD 2026 已经非常明确地占据：

\[
Continue/Stop
\]

甚至作者自己明确提出 POMDP 是下一步。

所以如果直接：

> “把 KDD 的 MDP 改成 POMDP，然后换 UMER。”

创新风险很大。

可以在后期加入 evidence sufficiency / uncertainty，但不应该成为论文最核心的新问题。

---

## 11. 与现有 UMER 的衔接

UMER 当前输入本来就有：

- propagation tree；
- node semantics；
- node/user features；
- depth；
- relative time；
- graph branch；
- DeBERTa textual branch。

因此真正需要改变的不是整套模型，而是把原来的：

\[
G_{240h}\rightarrow UMER\rightarrow Label
\]

变成：

\[
G_t\rightarrow UMER\ representation
\rightarrow Dynamic\ Evidence\ Selector
\rightarrow LLM
\]

即把 UMER 的角色部分改变为：

\[
\boxed{\text{Social-context encoder / evidence selector}}
\]

而不是再让 LLM 给 UMER 增加一个辅助特征。

---

## 12. 与此前 Round048 负结果的区别

Round048 是：

\[
Qwen\ stance/role/salience
\rightarrow
4\ replies
\]

结果认知覆盖更丰富，但 fidelity 没有可靠优于时间/哈希 baseline，因此失败。

所以新的 selector **不能再次依赖 LLM teacher 产生 salience**。

应该反过来：

\[
\boxed{
UMER/Graph\ model
\rightarrow
Evidence\ selection
\rightarrow
LLM
}
\]

LLM 是 evidence consumer，不是 evidence selector。

这既避开了已有负结果，也符合当前文献留下的问题。

---

## 13. 实验设计上的潜在要求

如果最终真的走这条线，只在 PHEME 上：

\[
1h/3h/6h/12h
\]

做几个时间切片，论文仍然容易被质疑：

> 又在旧 benchmark 上定义一种 early protocol。

因此 **DAWN 式 temporality-aware evaluation** 应该成为强制要求。

至少需要比较：

\[
\text{Random/legacy split}
\]

和：

\[
\text{Chronological split}
\]

并观察：

\[
SePro
\]

\[
LLM-only
\]

\[
UMER
\]

\[
Dynamic-context\ LLM
\]

在真正禁止未来知识之后分别下降多少。

这样研究问题可以从：

> “我们又提出了一个 early detector。”

升级为：

> **“现有 LLM + social-context 方法在严格时间条件下是否仍成立，以及如何修复其动态 context 缺陷。”**

---

## 14. 当前阶段结论

这一轮之后，已经没有必要继续泛化搜索“动态 + LLM”。

真正值得继续验证的问题已经收敛到：

\[
\boxed{
\textbf{Temporally Causal Dynamic Social-Context Refinement for LLM Rumor Detection}
}
\]

它不是：

- 加时间 embedding；
- 再训练一个动态图 GNN；
- 决定什么时候调用 LLM；
- 用 LLM 给 UMER 增强 feature；
- 把整棵传播树直接塞给 LLM。

而是研究：

> **在每个真实时间点，从当时已经形成的传播图中动态找到“最值得 LLM 阅读的社会证据”。**

这一研究空白由几个主流工作的边界自然围出来：

- **SePro：会选 context，但不是动态；**
- **KDD EARD：是动态，但没有传播结构；**
- **DCAGTN：有动态传播结构，但没有 LLM；**
- **DAWN：有严格时间协议，但没有 LLM；**
- **K-FLARE：解决无 context，而不是 context 演化。**

因此截至当前检索，这个候选是目前最站得住脚的学术定位。

下一步真正需要验证的是两个现实问题：

1. **现有数据是否足以构造严格动态传播快照；**
2. **UMER 的现有表示是否真的能选出比 SePro 静态 attention/top-k 更好的动态 evidence。**

如果这两个基础假设不成立，这个方向应及时淘汰，而不是继续堆模型。
