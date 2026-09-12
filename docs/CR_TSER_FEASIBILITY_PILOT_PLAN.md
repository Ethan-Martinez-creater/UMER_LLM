# CR-TSER Feasibility Pilot Plan

## 1. Research decision

This document freezes the next-stage research design after the completed TC-DSCR line.

The new direction is provisionally named:

# CR-TSER
## Cross-Reader Robust Temporal Social Evidence Refinement

The central research question is:

> Given a temporally evolving social-propagation graph, can structurally valid social-evidence interventions reveal utility signals that are both predictable from propagation structure/time and sufficiently stable across heterogeneous LLM readers to support context selection for a previously unseen reader?

The pilot is **not** intended to prove a final SOTA system. It must first establish that the scientific premises required for a publishable CR-TSER method actually hold.

The pilot has four mandatory questions:

- **P1 — Reader heterogeneity:** Do different LLM readers assign meaningfully different help/harm effects to the same social evidence?
- **P2 — Structural increment:** Do propagation structure and time provide utility-prediction signal beyond text and simple relevance?
- **P3 — Shared utility:** Is there a non-trivial reader-invariant utility component, rather than purely reader-specific behavior?
- **P4 — Unseen-reader transfer:** Can a multi-reader robust selector outperform a strong simple compression baseline on a completely unseen reader under the same social-token budget?

If these premises do not hold, CR-TSER must stop. The execution agent must not automatically redesign the method.

---

# 2. Literature-grounded design boundary

The design deliberately avoids several directions that are already crowded.

General context-sufficiency classifiers are not sufficient novelty: ICLR 2025 formalized "sufficient context"; NeurIPS 2025 showed lightweight latent sufficiency signals; ACL 2026 S2G-RAG uses a dedicated sufficiency judge.

Generic LLM-feedback utility reranking is also crowded: Uplift-RAG, SCARLet, utility-focused LLM annotation, RRPO, and 2026 utility-oriented retrieval work directly optimize evidence utility for downstream LLMs.

Reader-specific utility itself is not enough: recent work shows that the same evidence can help one reader and harm another, and that reader preference similarity does not reliably imply intervention transfer.

The remaining research space is therefore defined by the **joint** setting:

1. evidence is embedded in a **social propagation tree**, not a collection of independent passages;
2. interventions must respect **reply/parent/path/subtree structure**;
3. utility is conditioned on **real evidence arrival time**;
4. utility is measured across **multiple frozen LLM readers**;
5. the final selector is evaluated on a **reader whose intervention labels never participate in training**.

This joint setting, rather than any single component, is the novelty hypothesis to be tested.

---

# 3. Relationship to UMER and TC-DSCR

The new work should not preserve UMER merely for continuity. It should preserve only components that remain scientifically justified.

## 3.1 Components retained

### A. Normalized event representation

Reuse the existing normalized causal event abstraction:

```text
event_id
label
source_id
source_timestamp

nodes:
    node_id
    parent_id
    timestamp
    text
    original_order
    status
```

### B. Temporal snapshot protocol

Reuse the strict rule:

\[
V_t=\{v_i\mid timestamp_i \le t\}.
\]

No future reply, edge, text, subtree or intervention label may enter an earlier snapshot.

### C. Reply–Parent Evidence Unit

Keep the current atomic textual evidence unit:

```text
Reply text
+
Parent text
```

The reply node is the graph identity of the evidence unit.

### D. Frozen multilingual semantic encoder

Reuse:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
384D
```

as the frozen text feature extractor and semantic-relevance baseline.

### E. Prompt/parser/cache infrastructure

Reuse the existing reader-prompt, parsing, cache, deterministic inference and verifier design where possible.

### F. Evaluation discipline

Reuse:

- event-level split discipline;
- all-event accounting;
- manifest freezing;
- paired bootstrap preserving multiplicity;
- no test-time tuning;
- protocol verifiers.

## 3.2 Components not retained as the new core

The following components are **not** part of the CR-TSER core method:

```text
1021D positional adjacency signature
old CausalSocialEncoder as primary representation
Static Proxy classifier
novelty + persistence Dynamic Memory
MF-TSR
MS-TSR sufficiency objective
teacher-KL objective
```

They remain historical baselines/diagnostics only.

The reason is methodological:

- the fixed adjacency signature is tied to a bounded positional graph representation and is poorly suited to a new, potentially much larger Weibo22 graph;
- the Proxy/sufficiency line has already been experimentally shown not to transfer reliably to an independent LLM reader;
- the old Dynamic objectives have already been falsified.

## 3.3 Legacy continuity baseline

For **PHEME**, the frozen final TC-DSCR Static Utility score may be evaluated as a legacy baseline/feature because a clean checkpoint already exists.

For **Weibo22**, do **not** zero-shot reuse a Ma-Weibo Static Utility checkpoint and do not call it valid utility.

The scientific continuity is:

```text
UMER propagation modeling
    ↓
TC-DSCR causal social evidence refinement
    ↓
TC-DSCR reveals Proxy–LLM sufficiency mismatch
    ↓
CR-TSER directly studies real reader intervention utility
```

---

# 4. Pilot datasets

## 4.1 Primary dataset — Weibo22

Weibo22 is mandatory for the main pilot decision.

The official KPG release provides text files plus graph-processing scripts. The associated CUHK dataset description reports 2,087 rumor and 2,087 non-rumor source events and detailed propagation records.

Strict temporal CR-TSER requires true per-node timestamps. The public KPG processing script clearly exposes parent-child graph relations, but the processed graph script itself does not establish that every released record contains a usable original timestamp.

Therefore Weibo22 has a mandatory **P0 data feasibility gate**.

Required raw fields:

```text
event/source ID
binary rumor label
source text
source timestamp
reply/repost text
reply/repost timestamp
current node ID
parent node ID
```

`original_order` may be used only as a deterministic tie-break. It must never substitute for time.

If true per-node timestamps and parent relations can be reconstructed:

```text
WEIBO22_TEMPORAL_READY
```

Otherwise:

```text
WEIBO22_TEMPORAL_UNAVAILABLE
STOP
```

No pseudo-time based on row/node order is allowed.

## 4.2 Secondary dataset — PHEME

PHEME is retained as:

- an English-language comparison;
- continuity with the existing TC-DSCR study;
- a secondary test of dataset dependence.

PHEME cannot by itself satisfy the pilot GO decision.

## 4.3 Ma-Weibo

Ma-Weibo is not a primary CR-TSER pilot dataset.

It may be used only for engineering sanity checks or legacy comparisons and must not determine the GO/NO-GO decision.

---

# 5. Pilot event split

Use seed:

```text
7319
```

For each dataset, after viability filtering, construct event-disjoint subsets:

```text
foundation_train: 80 events
utility_train:    50 events
utility_dev:      15 events
utility_eval:     25 events
```

Total:

```text
170 events per dataset
```

Sampling should be approximately label-balanced whenever possible.

All remaining events are untouched.

`foundation_train` is reserved for optional/legacy representation baselines and receives no required LLM intervention labels.

`utility_train` trains utility predictors.

`utility_dev` controls early stopping using training-reader utility only.

`utility_eval` is final pilot feasibility evaluation; no hyperparameter adjustment is allowed after its outcomes are inspected.

---

# 6. Temporal cutoffs

Use exactly:

```text
15m
1h
6h
```

For every event, attempt all three.

Zero-reply snapshots are retained in the audit but do not generate intervention rows.

---

# 7. Canonical social-evidence context

Do not define the reference context with Proxy, MF-TSR or MS-TSR.

Define:

# SRC — Semantic Reference Context

For each snapshot:

1. create one Reply–Parent unit per visible non-source reply;
2. use frozen 384D MiniLM;
3. compute:
   \[
   rel_i=\cos(e_{reply_i},e_{source});
   \]
4. rank descending;
5. tie-break by earlier timestamp, then stable order;
6. greedily pack complete units under:
   \[
   B_{ref}=1024
   \]
   canonical social-evidence tokens.

The canonical budget tokenizer for all readers is:

```text
Qwen/Qwen3-8B tokenizer
```

Actual reader-specific prompt tokens are recorded separately.

This makes intervention contexts reader-independent, gold-independent and Proxy-independent.

---

# 8. Frozen LLM readers

Freeze exactly:

```text
R1 = Qwen/Qwen3-8B
R2 = zai-org/glm-4-9b-chat-hf
R3 = internlm/internlm3-8b-instruct
```

Before experiments record:

```text
local path
model ID
weight hash
tokenizer hash
dtype
transformers version
device config
chat-template hash
```

No substitution after intervention generation starts.

If any reader cannot be deployed reliably, P0 fails and execution stops.

All are frozen; no LoRA/fine-tuning/gradients.

Qwen thinking is disabled. InternLM uses normal/non-deep-thinking mode.


# 9. Reader task and label scoring

Do not use generated confidence as the primary utility signal.

The fixed task tells the reader:

```text
Use only the source post and supplied observed social evidence.
Do not add external facts.

Choose:
A = RUMOR
B = NON_RUMOR
```

PHEME text stays English; Weibo22 stays Chinese.

Use teacher-forced sequence scoring for candidate continuations `"A"` and `"B"`.

For candidate \(c\):

\[
s_r(c\mid C)=\sum_k\log p_r(c_k\mid prompt(C),c_{<k}).
\]

Normalize:

\[
p_r(A\mid C)=
\frac{\exp s_r(A)}
{\exp s_r(A)+\exp s_r(B)}.
\]

Prediction:

\[
\hat y_r(C)=\arg\max_{c\in\{A,B\}}s_r(c\mid C).
\]

Preflight records tokenization of A/B for every reader.

---

# 10. Intervention utility definition

Let \(C_{ref}\) be SRC. For removal group \(A\):

\[
\boxed{
u_r(A)=
p_r(y\mid C_{ref})
-
p_r(y\mid C_{ref}\setminus A)
}
\]

where \(y\) is the ground-truth label.

Gold is allowed only to construct utility supervision on pilot data. It is never a selector input.

Interpretation:

- positive: A helped support the correct decision;
- negative: A hurt the correct decision;
- near zero: little measured effect.

Also store:

```text
prediction_before
prediction_after
correct_before
correct_after
label_flip
gold_probability_before
gold_probability_after
```

## 10.1 Tri-class label

Correctness transitions override thresholds:

```text
correct -> wrong = HELPFUL
wrong -> correct = HARMFUL
```

Otherwise:

```text
u >= +0.05 = HELPFUL
u <= -0.05 = HARMFUL
else = NEUTRAL
```

Threshold 0.05 is frozen.

---

# 11. Intervention family

## I0 — Base

`C_ref`.

## I1 — Atomic Reply–Parent removal

Remove every evidence unit once.

These atomic labels train the utility model.

## I2 — Parent–child pair

Among selected reply nodes directly connected by an edge, enumerate pairs and choose the pair with highest combined SRC semantic relevance. Remove both.

If none:

```text
NO_PARENT_CHILD_PAIR
```

## I3 — Matched non-adjacent pair

Match I2 using only pre-reader features:

- not parent-child;
- neither is ancestor of the other;
- combined token cost within ±20% if possible;
- mean depth difference ≤1 if possible.

Deterministic seed 7319.

## I4 — Selected-subtree intervention

Find a selected reply node with at least two selected nodes in its current visible subtree.

Choose the root with the largest selected-descendant count; ties use earlier timestamp.

Remove all selected units in that subtree, capped to the earliest 5 selected units.

If no valid group:

```text
NO_SUBTREE_INTERVENTION
```

## I5 — Matched disconnected group

For I4 size \(k\), choose \(k\) units not all in one ancestor subtree and match total token cost ±20% where possible.

Reader outcomes cannot influence control matching.

Maximum per snapshot:

```text
1 base
+ N atomic removals
+ up to 4 structured/control interventions
```

If `N > 20`, only the top-20 SRC units by semantic relevance receive intervention labels. Record the cap.

---

# 12. Structural interaction

For group \(A\):

\[
Interaction_r(A)
=
u_r(A)
-
\sum_{e_i\in A}u_r(\{e_i\}).
\]

Edge interaction gap:

\[
\Delta_{edge}
=
E|I(parent-child)|
-
E|I(matched-nonadjacent)|.
\]

Subtree interaction gap:

\[
\Delta_{subtree}
=
E|I(subtree)|
-
E|I(matched-disconnected)|.
\]

Bootstrap by event.

This is the pilot's direct test that social-propagation structure creates non-additive reader effects beyond generic passage interactions.

---

# 13. Proposed structural model — BiTTE

# BiTTE
## Bidirectional Temporal Tree Encoder

The old fixed 1021D adjacency signature is not used by the proposed model.

## 13.1 Raw node input

Semantic:

\[
e_i\in\mathbb R^{384}
\]

from frozen MiniLM.

Ten current-snapshot structural/time scalars:

1. `depth_norm = min(depth,20)/20`
2. `child_count_norm = log1p(children)/log1p(N)`
3. `degree_norm = log1p(degree)/log1p(N)`
4. `subtree_size_norm = log1p(subtree_size)/log1p(N)`
5. `sibling_count_norm = log1p(siblings)/log1p(N)`
6. `is_source_child`
7. `is_leaf`
8. `elapsed_norm = log1p(elapsed)/log1p(cutoffSeconds)`
9. `parent_lag_norm = log1p(max(t_i-t_parent,0))/log1p(cutoffSeconds)`
10. `arrival_rank = rank/(N-1)`; use 0 if N=1.

All are computed strictly within \(G_t\).

## 13.2 Input projection

Semantic branch:

```text
384 -> 256
LayerNorm
GELU
```

Structure branch:

```text
10 -> 64 -> 64
GELU
LayerNorm
```

Concatenate 320D:

```text
320 -> 256
LayerNorm
GELU
Dropout(0.1)
```

yielding \(h_i^{(0)}\in R^{256}\).

## 13.3 Message passing

Use exactly two layers.

\[
m_i^{parent}=
W_p h_{parent(i)}
\]
or zero for source.

\[
m_i^{child}
=
W_c Mean_{j\in children(i)}h_j
\]
or zero if leaf.

\[
m_i^{self}=W_s h_i.
\]

\[
\tilde h_i
=
GELU(m_i^{self}+m_i^{parent}+m_i^{child})
\]

\[
h_i'=
LayerNorm(h_i+Dropout(\tilde h_i,0.1)).
\]

Each layer has separate \(W_s,W_p,W_c\in R^{256\times256}\).

No Transformer/GAT/GRU/RL is added to the pilot.

---

# 14. Atomic utility representation

For evidence reply i:

```text
h_i   = BiTTE reply state
h_src = source state
h_ctx = mean state of reply nodes selected in C_ref
```

Construct:

\[
z_i=[
h_i;
h_{src};
h_{ctx};
h_i\odot h_{ctx};
|h_i-h_{ctx}|;
q_i
]
\]

where \(q_i\) has six scalars:

```text
semantic cosine(reply,source)
canonical token cost / 1024
SRC rank percentile
log cutoff-time scalar
C_ref selected-count / 20 clipped
C_ref token utilization / 1024
```

Dimension:

```text
1286
```

Project:

```text
1286 -> 256 -> 256
GELU
Dropout(0.1)
LayerNorm
```

to \(z_i^U\in R^{256}\).

---

# 15. Shared + reader-residual utility

For two training readers:

\[
\hat u_r(i)=\mu_i+\delta_{r,i}.
\]

Shared head:

```text
256 -> 128 -> 1
GELU
```

gives \(\mu_i\).

Each training reader receives a learned 16D embedding.

Residual head:

```text
[256 + 16] -> 128 -> 1
GELU
```

gives \(\delta_{r,i}\).

No embedding exists for the held-out reader.

Auxiliary sign head:

```text
272 -> 128 -> 3
```

for HELPFUL/NEUTRAL/HARMFUL.

---

# 16. Loss and training

Reader regression:

\[
L_{reader}=SmoothL1(\hat u_r,u_r).
\]

Shared target:

\[
\bar u_i=(u_{r_a,i}+u_{r_b,i})/2.
\]

\[
L_{shared}=SmoothL1(\mu_i,\bar u_i).
\]

Sign classification:

\[
L_{sign}=CE(\hat c,c).
\]

Residual shrink:

\[
L_{resid}=Mean(\delta^2).
\]

Total:

\[
\boxed{
L
=
1.0L_{reader}
+
0.5L_{shared}
+
0.5L_{sign}
+
0.01L_{resid}
}
\]

Training:

```text
AdamW
lr = 1e-3
weight_decay = 1e-4
batch = 128 reader-intervention rows
max_epochs = 100
patience = 10
grad_clip = 1.0
seeds = 7319, 7320, 7321
```

Early stopping:

```text
mean utility_dev Spearman across the two training readers
```

Fallback if undefined:

```text
negative validation SmoothL1
```

No hyperparameter grid.

---

# 17. Predictor baselines

## B0 — Text Only

Inputs:

```text
reply semantic 384
source semantic 384
semantic cosine
token cost
context size
cutoff scalar
```

No graph.

## B1 — Text + Scalar Structure/Time

B0 plus the ten fixed structural/time scalars.

No message passing.

## B2 — Legacy TC-DSCR Diagnostic

PHEME only:

```text
frozen CausalSocialEncoder representation
frozen Static Utility score
```

This is continuity analysis, not a primary Weibo22 gate.

## B3 — BiTTE

Full proposed BiTTE + shared/residual heads.

Primary predictor task:

```text
HELPFUL / NEUTRAL / HARMFUL
```

Report:

```text
Macro-F1
per-class F1
AUROC helpful-vs-rest
AUROC harmful-vs-rest
Spearman continuous utility
MAE
active-intervention sign accuracy
```

---

# 18. Reader heterogeneity

An atomic intervention is active for reader r if:

```text
abs(u_r) >= 0.05
or correctness changes
```

Among interventions active for both reader pair (a,b):

\[
Disagree(a,b)=P[sign(u_a)\neq sign(u_b)].
\]

Report all 3 pairs and macro mean, plus utility Spearman and sign contingency.

---

# 19. Shared utility analysis

Compute descriptive:

\[
SharedRatio
=
\frac{Var(\mu)}
{Var(\mu)+Mean_r Var(\delta_r)}.
\]

Also compare:

```text
shared-only prediction
shared+residual
single-reader predictor
```

This is descriptive; do not call it a formal causal variance decomposition.

---

# 20. Leave-one-reader-out

Exactly three rotations:

```text
1:
train Qwen + GLM
hold InternLM

2:
train Qwen + InternLM
hold GLM

3:
train GLM + InternLM
hold Qwen
```

Held-out reader labels:

```text
not loaded during training
not used for early stopping
not used for selection
```

They become visible only after checkpoints and evidence subsets are frozen.

Use three training seeds and average predictions. Never choose the best seed.


# 21. Robust selection

For evidence i, average seed predictions separately for each training reader.

Define:

\[
s_i^{robust}
=
\min(\hat u_{r_a,i},\hat u_{r_b,i}).
\]

Density:

\[
d_i^{robust}
=
s_i^{robust}/\max(TokenCost_i,1).
\]

The pilot only reranks units already in C_ref; it does not retrieve outside C_ref.

Primary compressed target:

\[
B_{pilot}
=
\lfloor0.50\times Tokens(C_{ref})\rfloor.
\]

Packing:

1. rank;
2. scan;
3. add complete unit if total remains ≤ target;
4. otherwise skip and continue.

Always continue packing even when scores are negative; this keeps compression-budget comparisons controlled.

Prompt presentation is chronological.

---

# 22. Selection arms

For every utility_eval snapshot and held-out reader:

### S0 — SRC Full
Original reference context.

### S1 — Random Token-Matched
Deterministic seed 7319 under the 50% budget.

### S2 — Semantic Token-Matched
MiniLM relevance ranking under the same budget.

### S3a / S3b — Single-Reader Utility
One selector trained on each of the two training readers separately.

### S4 — Shared-Only Utility
Rank by \(\mu_i/token_i\).

### S5 — Cross-Reader Robust
Rank by worst-case training-reader utility density.

Primary proposed selector = S5.

### S6 — Legacy TC-DSCR Utility-TM
PHEME only; diagnostic.

---

# 23. Held-out reader evaluation

After subsets are frozen, score them using the held-out reader.

Report:

```text
Accuracy
Macro-F1
Rumor-F1
Non-rumor F1
mean/median social tokens
```

Primary P4 comparison:

```text
S5
vs
best simple baseline among S1 and S2
```

Also report S3a/S3b, S4, and SRC Full.

---

# 24. Statistical protocol

All intervals:

```text
event-level paired bootstrap
10000 iterations
seed 7319
```

All cutoffs/interventions of a sampled event move together.

Multiplicity must be preserved.

Structured-interaction bootstrap includes only events with valid structured/control matched groups.

---

# 25. Pre-registered feasibility gates

## P0 — Data + reader readiness

Weibo22 must have:

```text
source text
reply text
source timestamp
per-node timestamp
parent IDs
labels
```

No pseudo-time.

All 3 readers must load and deterministically score A/B candidates.

Pass:

```text
P0_PASS
```

Else STOP.

---

## P1 — Reader heterogeneity

Primary: Weibo22 utility_eval.

Mean pairwise sign disagreement among jointly active atomic interventions:

\[
\boxed{MeanDisagreement\ge0.10}
\]

and at least 2/3 reader pairs individually:

\[
Disagreement\ge0.05.
\]

---

## P2 — Structured social interaction

Primary: Weibo22.

\[
\Delta_{edge}
=
E|I(parent-child)|
-
E|I(matched-nonadjacent)|.
\]

Pass if:

\[
\boxed{\Delta_{edge}\ge0.02}
\]

and event-bootstrap 95% CI lower bound > 0.

Subtree-vs-disconnected is confirmatory but not mandatory because availability may be lower.

---

## P3 — Structural utility prediction increment

Primary: Weibo22 utility_eval.

Compare B3 against strongest B0/B1.

Pass if:

\[
\boxed{
MacroF1(B3)-\max(MacroF1(B0),MacroF1(B1))
\ge0.02
}
\]

and event-bootstrap 95% CI lower bound >0.

B3 continuous-utility Spearman must also not be worse than the strongest baseline.

---

## P4 — Unseen-reader transfer

For each LORO rotation:

\[
\Delta_r =
MF1(S5)-MF1(bestSimple_{S1/S2}).
\]

Pass if:

1.
\[
Mean(\Delta_r)\ge+0.01
\]

2. at least 2/3 rotations have \(\Delta_r>0\);

3. no rotation has:
\[
\Delta_r<-0.005;
\]

4. S5 obeys the same maximum token target.

---

## PHEME secondary condition

Require:

\[
Mean\Delta_{S5-bestSimple}\ge-0.005.
\]

PHEME alone cannot produce FULL_GO.

---

# 26. Final decision

## FULL_GO

Requires:

```text
P0 PASS
P1 PASS
P2 PASS
P3 PASS
P4 PASS
PHEME non-catastrophic
```

Recommendation:

```text
START_FULL_CR_TSER_METHOD_DEVELOPMENT
```

## PARTIAL_GO

Strong evidence exists but one practical boundary is narrowly missed.

Recommendation:

```text
STOP_FOR_RESEARCH_REVIEW
```

## NO_GO

Any core premise fails:

```text
Weibo22 temporal unavailable
reader heterogeneity weak
structured interaction absent
BiTTE has no structural increment
robust selection fails unseen-reader transfer
```

Recommendation:

```text
STOP_CR_TSER
```

No automatic redesign.

---

# 27. Why the gates are joint

A publishable direction requires:

\[
\boxed{
Reader\ Heterogeneity
\rightarrow
Structured\ Social\ Interaction
\rightarrow
Predictable\ Structural\ Utility
\rightarrow
Unseen\ Reader\ Transfer
}
\]

Reader heterogeneity alone reproduces reader-specific utility literature.

Structure alone is graph analysis.

Utility prediction alone is generic RAG utility learning.

Cross-reader transfer without structural increment would not establish a social-network-specific contribution.

---

# 28. Package layout

Create a new package:

```text
project/cr_tser/
├── __init__.py
├── config/
│   └── pilot_config.py
├── data/
│   ├── weibo22_adapter.py
│   ├── pilot_split.py
│   ├── snapshot_bridge.py
│   └── structural_stats.py
├── intervention/
│   ├── evidence_units.py
│   ├── semantic_reference.py
│   ├── intervention_generator.py
│   └── interaction_controls.py
├── readers/
│   ├── base_reader.py
│   ├── sequence_scorer.py
│   ├── qwen_reader.py
│   ├── glm_reader.py
│   └── internlm_reader.py
├── models/
│   ├── bitte.py
│   ├── text_baseline.py
│   ├── scalar_structure_baseline.py
│   ├── utility_heads.py
│   └── robust_selector.py
├── training/
│   ├── utility_dataset.py
│   ├── train_utility.py
│   └── checkpointing.py
├── evaluation/
│   ├── heterogeneity.py
│   ├── structural_interaction.py
│   ├── utility_prediction.py
│   ├── unseen_reader.py
│   └── bootstrap.py
└── tests/
```

Do not add CR-TSER as another old `dynamic_v4` module.

---

# 29. Existing modules to reuse

Where compatible, import rather than copy:

```text
project/tcdscr/data/normalized_schema.py
project/tcdscr/data/pheme_adapter.py
project/tcdscr/data/snapshot_builder.py
project/tcdscr/data/semantic_encoder.py
project/tcdscr/data/text_cleaning.py

project/tcdscr/llm/reader_prompt.py
project/tcdscr/llm/reader_parser.py
```

Do not modify old TC-DSCR result artifacts.

---

# 30. P0 implementation and stop point

Before full model implementation, build/run only:

```text
Weibo22 raw-field audit
Weibo22 normalized-adapter smoke
three reader load/hash audit
A/B sequence-score sanity
```

Outputs:

```text
results/cr_tser/p0/
├── weibo22_audit.json
├── weibo22_audit.md
├── reader_audit.json
├── label_scoring_sanity.json
└── P0_READINESS.md
```

Weibo22 audit must report:

```text
event count
label distribution
source-text coverage
reply-text coverage
timestamp coverage
parent coverage
duplicate IDs
negative timestamps
child earlier than parent count
unresolvable parent rate
events with >=1 valid reply
events viable at 15m/1h/6h
```

Reader sanity uses at least 20 fixed examples per model, each scored twice.

Prediction identity must be 100%.

After P0:

```text
COMMIT
PUSH
STOP
```

Research approval is required before expensive labels.

---

# 31. Manifest freeze

After P0 approval, create:

```text
results/cr_tser/manifests/
├── event_split.json
├── snapshot_manifest.jsonl
├── intervention_manifest.jsonl
└── hashes.json
```

Manifests become immutable after the first reader utility call.

---

# 32. Utility-label cache

Store one row per:

```text
dataset
event_id
cutoff
reader
intervention_id
```

including:

```text
base_context_hash
intervened_context_hash
reader_hash
prompt_hash

score_A
score_B
p_rumor
p_nonrumor

gold
utility

prediction_before
prediction_after
correctness_before
correctness_after

intervention_type
affected_reply_ids
```

Cache under:

```text
results/cr_tser/utility_labels/
```

Never recompute identical hashes.

---

# 33. Training order

For every dataset and reader-holdout rotation:

1. load utility_train labels for two training readers;
2. train B0/B1/B3 with seeds 7319/7320/7321;
3. early stop only with utility_dev labels for the same two readers;
4. freeze checkpoints;
5. evaluate utility predictor on utility_eval;
6. average three seed predictions;
7. construct all selection arms;
8. freeze evidence subsets;
9. only then expose held-out-reader utility_eval scores for transfer evaluation.

---

# 34. Verifier

Add:

```text
scripts/cr_tser_verify_pilot.py
```

It must check:

### Data

```text
all split events disjoint
no future node in snapshot
true timestamp used
no original_order substitution
structured intervention valid in G_t
```

### Reader leakage

```text
held-out reader labels absent from train batches
absent from early stopping
absent from context selection
all reader hashes frozen
```

### Utility

```text
utility recomputable from cached probabilities
correctness transition override exact
neutral threshold ±0.05 exact
```

### Selection

```text
selected units subset of C_ref
target exactly floor(0.5*C_ref tokens)
no arm exceeds target
Reply–Parent atomic units preserved
prompt presentation chronological
```

### Statistics

```text
event-level bootstrap
multiplicity preserved
all gate thresholds exactly match plan
```

Required:

```text
issues = 0
```

---

# 35. Minimum tests

```text
test_weibo22_timestamp_not_original_order
test_snapshot_excludes_future_node
test_parent_precedes_child_or_flags_anomaly
test_src_budget_never_exceeded

test_sequence_score_A_B
test_utility_sign_definition
test_correct_to_wrong_is_helpful
test_wrong_to_correct_is_harmful

test_parent_child_pair_is_real_edge
test_nonadjacent_control_not_ancestor
test_subtree_group_is_valid_descendant_set

test_bitte_parent_message
test_bitte_child_mean_message
test_bitte_padding_independent
test_bitte_handles_large_tree_without_1021_signature

test_heldout_reader_not_in_training_batch
test_heldout_reader_not_in_early_stopping
test_heldout_reader_not_in_selection

test_robust_score_is_min_train_readers
test_selection_never_exceeds_50pct_target
test_selection_uses_only_reference_candidates

test_bootstrap_preserves_multiplicity
test_gate_logic_exact
```

---

# 36. Results layout

```text
results/cr_tser/
├── p0/
├── manifests/
├── utility_labels/
├── predictor/
│   ├── weibo22/
│   └── pheme/
├── intervention_analysis/
├── unseen_reader/
├── verifier/
├── CR_TSER_PILOT_REPORT.md
└── CR_TSER_PILOT_SUMMARY.json
```

---

# 37. Required report tables

### A — Reader heterogeneity

```text
dataset
reader pair
active interventions
sign disagreement
utility Spearman
```

### B — Structured interaction

```text
dataset
reader
edge |I|
non-edge |I|
delta
95% CI

subtree |I|
disconnected |I|
delta
95% CI
```

### C — Utility prediction

```text
B0 Text
B1 Scalar Structure-Time
B2 Legacy (PHEME)
B3 BiTTE

Macro-F1
Helpful F1
Harmful F1
Spearman
MAE
```

### D — Leave-one-reader-out selection

```text
held-out reader
SRC Full
Random-TM
Semantic-TM
Single-reader A
Single-reader B
Shared
Cross-reader Robust

Macro-F1
social tokens
```

### E — Gate summary

```text
P0
P1
P2
P3
P4
PHEME secondary
FINAL
```

---

# 38. Report structure

```markdown
# CR-TSER Feasibility Pilot

## Protocol Freeze
## P0 Data / Reader Readiness
## Dataset and Split Audit
## Reference Context Statistics
## Reader Heterogeneity
## Structured Intervention Effects
## Utility Prediction
## Shared vs Reader-Specific Utility
## Leave-One-Reader-Out Transfer
## PHEME Secondary Evidence
## GO / NO-GO Gates
## Verifier
## Final Recommendation
```

Final recommendation only:

```text
START_FULL_CR_TSER_METHOD_DEVELOPMENT
STOP_FOR_RESEARCH_REVIEW
STOP_CR_TSER
```

---

# 39. Execution checkpoints

## Checkpoint 1 — P0

Implement only:

```text
Weibo22 adapter/audit
reader wrappers
sequence-score sanity
```

Commit, push, STOP.

## Checkpoint 2 — Code Complete

After P0 approval implement:

```text
split
SRC
interventions
label cache
BiTTE
utility heads
baselines
LORO
selection
metrics
verifier/tests
```

Run only tests/smoke.

Commit, push, STOP.

Do not generate the full intervention dataset.

## Checkpoint 3 — Pilot Complete

Only after code approval:

```text
generate labels
train predictors
run unseen-reader evaluation
compute gates
```

Commit, push, STOP.

No post-hoc redesign.

---

# 40. Approximate workload

Per dataset:

```text
90 utility-labeled events
× 3 cutoffs
≈ 270 attempted snapshots
```

Two datasets:

```text
≈ 540 attempted snapshots
```

Each valid snapshot/reader:

```text
1 base
+ N atomic
+ <=4 structure/control groups
N <= 20
```

All reader calls are A/B sequence scoring, not long rationale generation.

Batch aggressively.

---

# 41. What FULL_GO justifies

If all gates pass, a full paper-scale project can be developed around:

1. a temporal social-evidence intervention benchmark;
2. shared + reader-specific utility modeling;
3. cross-reader robust social-context selection;
4. unseen-reader transfer;
5. Weibo22 as the primary modern Chinese propagation dataset.

At that point the paper is no longer “UMER plus an LLM module.”

It is a new study whose scientific lineage is:

```text
UMER
→ TC-DSCR
→ discovered Proxy–LLM mismatch
→ CR-TSER
```

---

# 42. What NO_GO means

P1 fail:
```text
reader-specific utility is not a sufficiently strong problem here
```

P2 fail:
```text
social graph dependence is not distinguishable from generic passage interaction
```

P3 fail:
```text
graph/time does not provide learnable incremental utility signal
```

P4 fail:
```text
utility does not transfer well enough to justify robust selection
```

In all cases:

```text
STOP
```

No CR-TSER V2 is created automatically.

---

# 43. Frozen scientific chain

The pilot attempts to establish:

\[
\boxed{
Reader\ Heterogeneity
\rightarrow
Structured\ Social\ Interaction
\rightarrow
Learnable\ Structural\ Utility
\rightarrow
Cross\text{-}Reader\ Transfer
}
\]

A publishable CR-TSER claim is not allowed until this chain is supported on Weibo22.

---

# 44. Literature basis

1. Hailey Joren et al. *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.* ICLR 2025.  
https://arxiv.org/abs/2411.06037

2. Roy Xie et al. *Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals.* NeurIPS 2025.  
https://proceedings.neurips.cc/paper_files/paper/2025/hash/3afa7f9b7ac8da474b3d915570d58291-Abstract-Conference.html

3. Minghan Li et al. *S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA.* ACL 2026.  
https://aclanthology.org/2026.acl-long.1185/

4. Changle Qu et al. *Uplift-RAG: Uplift-Driven Knowledge Preference Alignment for Retrieval-Augmented Generation.* Findings EMNLP 2025.  
https://aclanthology.org/2025.findings-emnlp.511/

5. Yilong Xu et al. *Training a Utility-based Retriever Through Shared Context Attribution for Retrieval-Augmented Language Models.* EMNLP 2025.  
https://aclanthology.org/2025.emnlp-main.33/

6. Hengran Zhang et al. *Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation.* EMNLP 2025.  
https://aclanthology.org/2025.emnlp-main.88/

7. Shi Zhou. *Preference Is Not Intervention: The Structure and Stability Boundaries of Reader-Specific Evidence Utility.* arXiv 2026.  
https://arxiv.org/abs/2608.17781

8. Chaoqun Cui et al. *Enhancing Rumor Detection Methods with Propagation Structure Infused Language Model.* COLING 2025.  
https://aclanthology.org/2025.coling-main.478/

9. Xingyu Peng et al. *Rumor Detection on Social Media with Temporal Propagation Structure Optimization.* COLING 2025.  
https://aclanthology.org/2025.coling-main.261/

10. Yusong Zhang et al. *Rumor Detection on Social Media with Reinforcement Learning-based Key Propagation Graph Generator.* WWW 2025.  
https://doi.org/10.1145/3696410.3714651

11. Shuzhi Gong et al. *Invariant Subgraphs for Cross-Domain Fake News Detection via Causal Disentanglement.* ICWSM 2026.  
https://doi.org/10.1609/icwsm.v20i1.42673

12. Hengran Zhang et al. *LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation.* arXiv 2025.  
https://arxiv.org/abs/2510.11358

13. KPG / Weibo22 official repository.  
https://github.com/kkkkk001/KPG

14. CUHK Research Data Management Development Fund — Weibo22/KPG description.  
https://www.lib.cuhk.edu.hk/en/research/data/rdm-funds-grants/rdmdf2022/

15. Qwen Team. *Qwen3-8B model card.*  
https://huggingface.co/Qwen/Qwen3-8B

16. Z.ai. *GLM-4-9B-Chat-HF model card.*  
https://huggingface.co/zai-org/glm-4-9b-chat-hf

17. InternLM. *InternLM3-8B-Instruct model card.*  
https://huggingface.co/internlm/internlm3-8b-instruct
