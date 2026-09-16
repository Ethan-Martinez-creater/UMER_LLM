# BCR-Utility Research Execution Plan V1
## Behaviorally Conditioned Reader-Specific Evidence Utility for Temporal Rumor Verification

**Repository baseline:** `518a821d8d839fb7857be99acb91fde71b80c846`  
**Historical CR-TSER status:** `CR_TSER_V2R1_FEASIBILITY = NO_GO`  
**New research namespace:** `bcr_v1`  
**Primary dataset:** Ma-Weibo  
**Secondary dataset:** PHEME  
**Execution style:** staged / fail-closed / approval after every major gate

---

# 0. Research decision

The previous CR-TSER line is closed and read-only.

Frozen historical result:

```text
P1 = PASS
P2 = FAIL
P3 = NOT RUN
P4 = NOT RUN

CR_TSER_V2R1_FEASIBILITY = NO_GO
reason = STRUCTURED_INTERACTION_GATE_NOT_SUPPORTED
```

The important retained finding is not the structural hypothesis. It is the
reader-dependent utility phenomenon:

```text
Ma-Weibo P1 mean sign disagreement = 0.4082
PHEME diagnostic mean disagreement = 0.3764
```

The new research line therefore changes the modeling emphasis from:

```text
propagation structure
    -> shared utility
    -> reader residual
```

to:

```text
evidence semantics
+
reader behavioral state
+
reader-evidence compatibility
+
time
    -> reader-specific helpful / neutral / harmful utility
```

Working name:

# BCR-Utility
## Behaviorally Conditioned Reader-Specific Evidence Utility

Central question:

> Can a lightweight, utility-label-free behavioral profile of an LLM reader
> predict which temporal social evidence will help or harm that reader,
> including when the reader was never seen during utility-model training?

The core target is **signed intervention utility**, not generic preference or
document ranking.

---

# 1. Literature-grounded boundary

This plan is explicitly constrained by four external findings.

## 1.1 LLM-specific utility is already a known phenomenon

`LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation`
(arXiv:2510.11358; latest 2026 revision) shows that utilitarian evidence is
model-specific and does not transfer cleanly across LLMs. It also identifies
readability/perplexity as one explanatory factor and reports that existing
utility-judgment methods still struggle to model truly model-specific utility.

Therefore this project must **not** claim novelty merely from:

```text
"different LLMs prefer different evidence"
```

That phenomenon is an input assumption, independently supported by our P1.

## 1.2 Preference similarity is not sufficient for intervention transfer

`Preference Is Not Intervention: The Structure and Stability Boundaries of
Reader-Specific Evidence Utility` (arXiv:2608.17781, Aug 2026) separates:

```text
evidence activity
ordinal preference
conditional signed direction
```

and shows that stable ordinal reader geometry does not by itself imply
cross-reader help/harm transfer.

Therefore BCR-Utility must evaluate **conditional signed utility directly**.
A ranking-only result is insufficient.

The same work reports that signed reader geometry is substantially stronger in
binary fact-checking than in open-ended QA. This motivates, but does not prove,
our rumor-verification hypothesis.

## 1.3 Utility, not relevance, is the target

Recent utility-centric retrieval work treats downstream contribution as
different from topical relevance. BCR-Utility therefore keeps semantic
relevance only as a baseline feature.

## 1.4 Novelty target

The intended contribution is the combination:

```text
unlabeled behavioral reader probes
+
reader-evidence compatibility
+
signed helpful/harmful intervention utility
+
completely unseen-reader transfer
+
temporal social misinformation evidence
```

The project must not collapse into a generic utility reranker, reader-ID
classifier, or relevance model.

---

# 2. Historical artifacts and immutability

The following namespaces are historical and permanently read-only:

```text
results/cr_tser/
results/cr_tser_v2/
results/cr_tser_v2r1/
```

Historical reference commit:

```text
518a821d8d839fb7857be99acb91fde71b80c846
```

Important frozen v2r1 inputs:

```text
Ma-Weibo utility cache:
6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3

PHEME utility cache:
773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15
```

Existing v2r1 atomic evidence coverage:

```text
Ma-Weibo:
2549 I1 atomic evidence keys
90 labelled events
258 event x cutoff pairs

PHEME:
2107 I1 atomic evidence keys
90 labelled events
261 event x cutoff pairs
```

Existing readers:

```text
qwen      = Qwen/Qwen3-8B
mistral   = mistralai/Mistral-7B-Instruct-v0.3
internlm  = internlm/internlm3-8b-instruct
```

Historical structured interventions I2-I5 may be read for historical comparison,
but they are **not** part of the BCR-Utility core supervision.

No new code may write into any CR-TSER result namespace.

---

# 3. New repository namespace

All new code must live outside the frozen CR-TSER research line.

Recommended layout:

```text
project/bcr_utility/
├── __init__.py
├── config/
│   └── protocol.py
├── data/
│   ├── historical_import.py
│   ├── atomic_manifest.py
│   └── confirmatory_split.py
├── probes/
│   ├── probe_manifest.py
│   ├── probe_contexts.py
│   └── fingerprint.py
├── features/
│   ├── evidence_features.py
│   ├── nli_features.py
│   ├── tokenizer_features.py
│   └── compatibility_features.py
├── readers/
│   ├── registry.py
│   └── adapters/
├── models/
│   ├── baselines.py
│   └── conditioned_utility.py
├── evaluation/
│   ├── reader_geometry.py
│   ├── utility_metrics.py
│   ├── bootstrap.py
│   └── unseen_reader.py
├── tests/
└── verifier.py

scripts/
├── bcr_build_protocol.py
├── bcr_import_historical_atomic.py
├── bcr_build_probe_manifest.py
├── bcr_run_reader_probes.py
├── bcr_extract_features.py
├── bcr_run_mechanism_pilot.py
├── bcr_preflight_readers.py
├── bcr_generate_atomic_labels.py
├── bcr_run_reader_transfer.py
├── bcr_build_confirmatory_split.py
└── bcr_verify.py

results/bcr_utility_v1/
```

Reuse CR-TSER code only through explicit, read-only imports when the scientific
contract is unchanged, e.g.:

```text
dataset adapters
causal snapshot construction
Reply-Parent evidence-unit construction
SRC construction
canonical Qwen tokenizer budget
teacher-forced A/B scorer
```

Every reused dependency must be recorded by source-file SHA256 in the BCR
protocol manifest.

Do not modify historical CR-TSER code merely to support BCR.

---

# 4. New research hypotheses

## H1 — Reader-conditioned signed utility is predictable

For an evidence unit `e`, reader `r`, and time `t`:

```text
u(e, r, t)
```

cannot be adequately predicted from reader-agnostic evidence features alone.

A behavioral representation of `r` should provide incremental predictive value.

## H2 — Behavioral reader profiles contain reusable information

A reader representation obtained without target utility labels should explain
part of reader-specific signed utility variation.

The representation must not be a reader ID lookup table.

## H3 — Unseen-reader transfer is possible

Given utility labels from training readers and only an unlabeled behavioral
fingerprint for a new reader, the model should improve signed utility prediction
for the unseen reader over reader-agnostic baselines.

## H4 — Harm susceptibility is a meaningful secondary target

Reader-conditioned features should help predict:

```text
HARMFUL vs non-HARMFUL
```

social evidence, not only continuous utility.

## H5 — Temporal sufficiency is optional downstream work

Only if H1-H3 pass may a later stage study reader-specific sufficiency or
abstention at 15m/1h/6h.

H5 is not part of the initial GO decision.

---

# 5. Utility target

Reuse only the already justified atomic removal definition.

For reader `r`, evidence unit `e`:

```text
u_r(e) =
p_r(gold | SRC)
-
p_r(gold | SRC without e)
```

Tri-class target remains:

```text
correct -> wrong = HELPFUL
wrong -> correct = HARMFUL

otherwise:
u >= +0.05 = HELPFUL
u <= -0.05 = HARMFUL
else = NEUTRAL
```

The 0.05 threshold remains historical/frozen for imported rows.

For newly generated readers, use exactly the same definition and scoring
semantics.

BCR does not use I2-I5 structured group labels as training targets.

---

# 6. Reader representation: behavioral fingerprint

The fingerprint must be:

```text
utility-label-free
fixed before unseen-reader evaluation
small
deterministic
available for a completely new reader
```

It must not use:

```text
target reader utility labels
target utility_eval outcomes
reader ID embedding
reader-specific fine-tuning
```

## 6.1 Probe source

Use only the historical `foundation_train` events, because they received no
required intervention utility labels in CR-TSER.

Build a frozen probe manifest from:

```text
Ma-Weibo foundation_train
PHEME foundation_train
```

Deterministic seed:

```text
7319
```

For each dataset, select 36 events:

```text
12 assigned to 15m
12 assigned to 1h
12 assigned to 6h
```

Selection should be approximately rumor/non-rumor balanced, but **gold labels
are used only to balance the probe manifest**, never as fingerprint features.

An event assigned to a cutoff must have at least one valid visible evidence unit
at that cutoff.

The probe event IDs and cutoff assignment are frozen before reader probing.

## 6.2 Probe contexts

For every probe snapshot build exactly four contexts:

```text
P0 = source only
P1 = source + highest semantic-relevance valid evidence unit
P2 = source + lowest semantic-relevance evidence unit among SRC-selected units
P3 = source + full SRC under the frozen 1024 canonical-token budget
```

Selection is reader-independent.

Do not use utility outcomes to choose probe evidence.

## 6.3 Probe outputs

Using the same teacher-forced A/B classification contract, record:

```text
logprob(A)
logprob(B)
normalized P(A)
normalized P(B)
signed A-B margin
absolute margin
binary entropy
predicted class
prompt tokens
```

For P1/P2/P3 also record relative to P0:

```text
delta margin
absolute delta margin
delta entropy
prediction flip
```

## 6.4 Fingerprint vector

Create deterministic summary features grouped by:

```text
dataset
cutoff
context type
```

At minimum include:

```text
mean / std absolute margin
mean entropy
mean signed margin

for P1/P2/P3:
mean delta margin
mean absolute delta margin
flip rate
mean delta entropy

tokenization:
mean prompt-token ratio to canonical Qwen token count
```

The raw per-probe response vector must also be retained for geometry analysis,
but the main model uses the compact summary fingerprint.

No dimension may depend on utility labels.

---

# 7. Evidence feature protocol

Evidence features must be divided into explicit tiers.

## 7.1 E0 — Reader-agnostic semantic/text features

For every atomic Reply-Parent evidence unit:

```text
MiniLM reply-source cosine
MiniLM parent-source cosine
MiniLM reply-parent cosine
SRC relevance
SRC rank percentile
canonical token cost
reply character length
parent character length
reply-parent character length
elapsed time
cutoff identity
```

Simple CR-TSER graph scalars may be included only in a named **control
baseline**, not in the proposed core model.

## 7.2 E1 — NLI / stance-proxy features

Freeze:

```text
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

unless deployment preflight fails.

This model is used only as a frozen feature extractor.

For source vs reply and source vs parent, record:

```text
entailment probability
neutral probability
contradiction probability
```

Also record reply vs parent NLI.

These are compatibility/content features, not utility supervision.

If the exact NLI model cannot be deployed reliably, STOP for approval; do not
autonomously replace it.

## 7.3 E2 — Tokenizer-only reader compatibility

Available without loading model weights after tokenizer deployment:

```text
reader token count
canonical token count
fragmentation ratio = reader_tokens / canonical_tokens
reply fragmentation
parent fragmentation
combined evidence fragmentation
```

This is allowed in the **zero-touch** unseen-reader setting.

## 7.4 E3 — Light-touch reader compatibility

Optional secondary setting, requiring reader forward passes:

```text
source-only A/B margin
source-only entropy
evidence NLL per token
conditional evidence NLL given source
source NLL
NLL gap
```

These features may improve prediction but make target-reader adaptation more
expensive.

Therefore two settings must always be reported separately:

```text
ZERO-TOUCH:
fingerprint + E0/E1 + tokenizer-only E2

LIGHT-TOUCH:
ZERO-TOUCH + E3
```

A light-touch success must never be reported as zero-touch success.

---

# 8. Proposed prediction model

The main proposed model must remain intentionally small.

## 8.1 Evidence encoder

Input:

```text
E0 + E1 + optional E2/E3
```

Use:

```text
feature normalization
Linear -> GELU -> Linear
d_e = 32
```

## 8.2 Reader encoder

Input:

```text
behavioral fingerprint
```

Use:

```text
feature normalization
Linear -> GELU -> Linear
d_r = 32
```

## 8.3 Compatibility interaction

Construct:

```text
z = [evidence_embedding,
     reader_embedding,
     evidence_embedding * reader_embedding]
```

Then two heads:

```text
continuous utility regression
3-class HELPFUL / NEUTRAL / HARMFUL classification
```

Keep total trainable parameter count `< 250k`.

No graph neural network.

No reader-ID embedding in the proposed unseen-reader model.

---

# 9. Baselines

All baselines are mandatory.

## B0 — Evidence-only

```text
E0/E1 -> utility/sign
```

No reader information.

This is the primary comparator.

## B1 — Evidence + simple time/structure

Adds CR-TSER scalar structure/time features.

Purpose:

```text
negative/control comparison after CR-TSER P2 NO-GO
```

It must not be presented as the proposed model.

## B2 — Reader-ID model

Reader one-hot / learned ID embedding.

This model is only an in-domain diagnostic because it cannot represent an
unseen reader.

Do not use B2 as the main unseen-reader comparator.

## B3 — Nearest-reader transfer

Choose the closest training reader in fingerprint space and use its predicted
utility behavior.

This tests whether simple fingerprint similarity is sufficient.

## B4 — ZERO-TOUCH BCR

Main proposed model:

```text
evidence + behavioral fingerprint + tokenizer compatibility
```

## B5 — LIGHT-TOUCH BCR

Secondary upper-cost model:

```text
B4 + reader forward-pass compatibility features
```

---

# 10. Metrics

## 10.1 Primary metric

```text
tri-class utility-sign Macro-F1
```

because the intended intervention object is:

```text
HELPFUL / NEUTRAL / HARMFUL
```

## 10.2 Secondary metrics

```text
continuous utility Spearman
MAE
HARMFUL-vs-rest AUPRC
HARMFUL-vs-rest F1
HELPFUL-vs-rest AUPRC
```

## 10.3 Disagreement-focused evaluation

Define a training-reader disagreement subset without using held-out reader
labels:

```text
training readers do not all assign the same active sign
```

Evaluate held-out reader prediction separately on this subset.

This directly targets the phenomenon established by CR-TSER P1.

## 10.4 Statistical unit

All confidence intervals and comparisons use:

```text
event-level bootstrap
10000 iterations
seed 7319
```

Never bootstrap individual utility rows independently.

---

# 11. Phase M0 — Protocol migration and historical bootstrap

### Purpose

Create the new research line without spending new LLM inference.

### Tasks

1. Create `project/bcr_utility/`, scripts and result namespace.
2. Freeze `bcr_v1` configuration.
3. Implement historical read-only importer.
4. Verify historical cache/manifests against frozen SHA256.
5. Extract **I1 atomic only** into a BCR bootstrap table.
6. Build reader-utility geometry diagnostics:
   - activity;
   - ordinal utility;
   - conditional signed direction;
   - event-level reader x evidence interaction.
7. Build the 72-item probe manifest:
   - 36 Ma-Weibo;
   - 36 PHEME;
   - 12/cutoff/dataset.
8. Implement but do not yet run real-reader behavioral probes.
9. Implement BCR verifier and tests.

### M0 outputs

```text
results/bcr_utility_v1/bootstrap/
├── historical_identity.json
├── atomic_index.json
├── geometry_diagnostic.json
└── probe_manifest.json
```

### M0 stop rule

After implementation/tests/bootstrap artifacts:

```text
commit
push
STOP
```

No model download.
No new LLM call.
No new utility labels.

---

# 12. Phase M1 — Three-reader mechanism pilot

This phase uses only the already-approved:

```text
Qwen3-8B
Mistral-7B-Instruct-v0.3
InternLM3-8B-Instruct
```

No reader expansion yet.

## M1-A — Probe execution

Run the frozen 72-item probe battery for all three readers.

Use the existing server model weights.

Compute fingerprints.

## M1-B — Feature extraction

For historical atomic rows, generate:

```text
E0
E1
E2
E3
```

Do not alter historical utility labels.

## M1-C — Three-reader leave-one-reader-out pilot

Exactly three rotations:

```text
train mistral + internlm -> hold qwen
train qwen + internlm    -> hold mistral
train qwen + mistral     -> hold internlm
```

Training events:

```text
utility_train = 50
```

Model selection:

```text
utility_dev = 15
```

One-shot mechanism gate:

```text
utility_eval = 25
```

The historical utility_eval split has been observed only through aggregate
CR-TSER P1/P2 statistics, so M1 is a **feasibility pilot**, not final
publication-grade confirmation.

## M1 primary GO gate

Compare B4 ZERO-TOUCH to B0 evidence-only on Ma-Weibo.

Require all:

```text
mean Delta Macro-F1 >= +0.03
event-bootstrap 95% CI lower bound > 0
at least 2 / 3 held-out readers have positive Delta Macro-F1
no held-out reader drops by more than 0.05 Macro-F1
```

Secondary support:

```text
mean continuous Spearman improvement > 0
HARMFUL AUPRC does not decrease by > 0.02
```

### M1 outcomes

**M1_FULL_GO**

ZERO-TOUCH passes the primary gate.

Proceed to reader-panel expansion.

**M1_CONDITIONAL_GO**

ZERO-TOUCH fails, but LIGHT-TOUCH B5 satisfies the same primary gate.

STOP for research review before proceeding because the scientific claim changes
from lightweight fingerprint transfer to per-evidence target-reader probing.

**M1_NO_GO**

Neither passes.

Freeze the research line and do not expand readers.

---

# 13. Phase M2 — Expand to six readers

Run only after M1_FULL_GO or explicit approval after M1_CONDITIONAL_GO.

Target six-reader panel:

```text
qwen_small:
Qwen/Qwen3-0.6B

qwen:
Qwen/Qwen3-8B

mistral:
mistralai/Mistral-7B-Instruct-v0.3

internlm:
internlm/internlm3-8b-instruct

phi:
microsoft/Phi-4-mini-instruct

gemma:
google/gemma-3-4b-it
```

Rationale:

```text
multiple model families
large/small capacity variation
multilingual coverage
all expected to fit sequentially on RTX 4090
```

No seventh model may be added autonomously.

## Existing resources

Already deployed and must not be redownloaded:

```text
Qwen3-8B
Mistral-7B-Instruct-v0.3
InternLM3-8B-Instruct
```

Historical server resource:

```text
Qwen3-0.6B
```

Its exact current server path must be discovered; do not guess.

If absent after an actual search, download the official checkpoint.

## New models

Deploy only:

```text
microsoft/Phi-4-mini-instruct
google/gemma-3-4b-it
```

unless already present.

Gemma access may require accepting Google's model terms. If access is blocked:

```text
STOP
notify user
do not substitute another model
```

## M2 preflight

Every reader must pass:

```text
exact model identity
weight/tokenizer/chat-template hash
BF16 or official full-precision supported dtype
CUDA load
A/B candidate token audit
tokenizer-native boundary audit
20 sanity prompts x 2
identity_rate = 1.0
finite logprobs
```

Use isolated compatibility overlays if necessary.

Do not mutate `DGPA` site-packages globally.

---

# 14. Phase M2 label expansion

The new direction only needs atomic utility.

For the three added readers:

```text
qwen_small
phi
gemma
```

generate:

```text
I0 base
I1 atomic removals
```

on the same historical BCR development events.

Do not generate:

```text
I2
I3
I4
I5
```

Reuse the old three readers' frozen I1 rows by exact verified import.

New BCR label namespace:

```text
results/bcr_utility_v1/utility_labels/
```

It must not point back into CR-TSER output for writing.

Expected atomic counts are pinned from the frozen manifests:

```text
Ma-Weibo = 2549 atomic evidence keys per reader
PHEME    = 2107 atomic evidence keys per reader
```

Exact generation count may be larger if base rows are stored separately; the
atomic count must remain exact.

---

# 15. Phase M3 — Six-reader behavioral geometry and transfer development

## M3-A — Probe all six readers

Run the exact same frozen probe manifest.

No probe may be changed after seeing new-reader behavior.

## M3-B — Fingerprint reliability

Evaluate split-half probe reliability.

For each reader:

```text
fingerprint from first half
fingerprint from second half
```

Check:

```text
own-reader similarity > cross-reader similarity
```

Also compute pairwise fingerprint distance matrix.

This is a diagnostic, not by itself evidence of intervention transfer.

## M3-C — Geometry comparison

Compute six-reader utility geometries:

```text
activity geometry
ordinal geometry
conditional signed geometry
```

Compare fingerprint distance with **signed utility geometry**.

Do not claim transfer from ordinal geometry alone.

## M3-D — Six-reader leave-one-reader-out

Exactly six rotations.

For each held-out reader:

```text
utility labels from held-out reader:
forbidden in training
forbidden in model selection
forbidden in normalization fitting

behavioral probe fingerprint:
allowed

tokenizer-only compatibility:
allowed

light-touch compatibility:
allowed only in B5
```

Train on five readers using utility_train.

Tune/early stop on utility_dev.

Evaluate once on utility_eval.

## M3 GO gate

B4 ZERO-TOUCH vs B0 evidence-only on Ma-Weibo:

```text
mean Delta Macro-F1 >= +0.03
event-bootstrap 95% CI lower > 0
>= 4 / 6 held-out readers have positive Delta Macro-F1
no reader regression worse than -0.05
```

Additional requirement:

```text
disagreement-subset mean Delta Macro-F1 > 0
```

Secondary:

```text
HARMFUL AUPRC mean change >= 0
continuous Spearman mean change > 0
```

If B4 fails but B5 passes, stop for research review.

Do not automatically redefine the core claim.

---

# 16. Phase M4 — Fresh confirmatory Ma-Weibo evaluation

This stage is mandatory for a publication-grade positive result.

The historical utility_eval split has already informed prior research decisions.
Therefore the final claim cannot rely only on that split.

## M4-A — Freeze untouched confirmatory events

From Ma-Weibo viable events never included in historical:

```text
foundation_train
utility_train
utility_dev
utility_eval
```

sample:

```text
30 new events
```

using a new frozen seed:

```text
20260916
```

Requirements:

```text
event-disjoint from all historical CR-TSER/BCR development events
approximately label-balanced
true timestamps
same 15m/1h/6h causal protocol
```

Before any new utility labels are generated, freeze:

```text
event IDs
snapshots
SRC
atomic evidence manifest
hashes
all model hyperparameters
all evaluation gates
```

No tuning after this point.

## M4-B — Generate confirmatory utility labels

Generate only:

```text
I0
I1 atomic
```

for all six readers.

Do not use confirmatory outcomes for training.

## M4-C — Frozen six-reader unseen-reader evaluation

Use the models/hyperparameters frozen after M3.

For each held-out reader, evaluate on the new events.

## Final primary GO gate

B4 ZERO-TOUCH vs B0:

```text
mean Delta Macro-F1 >= +0.03
95% event x reader bootstrap CI lower > 0
>= 4 / 6 readers positive
no reader below -0.05
```

Secondary confirmation:

```text
HARMFUL AUPRC mean change >= +0.02
or
HARMFUL F1 mean change >= +0.02

continuous utility Spearman mean change > 0
```

The primary GO decision is Macro-F1 based.

If the final confirmatory gate fails:

```text
BCR_UTILITY_V1 = NO_GO
```

Do not tune on confirmatory events.

---

# 17. Phase M5 — PHEME secondary replication

Only after Ma-Weibo final GO.

PHEME is not allowed to rescue a failed Ma-Weibo primary result.

Run the same frozen six-reader evaluation using:

```text
existing PHEME development utility rows
```

and, if sufficient untouched viable PHEME events exist, a fresh secondary
confirmatory set.

Report:

```text
cross-language direction
reader-specific harmful evidence behavior
zero-touch vs light-touch transfer
```

PHEME remains secondary.

---

# 18. Optional Phase M6 — Early sufficiency / abstention

Not authorized by this master plan automatically.

Only consider after BCR unseen-reader transfer is confirmed.

Potential question:

```text
At 15m / 1h / 6h, is the available evidence sufficient for this reader,
or should the system abstain / wait?
```

This requires a separate protocol and approval.

Do not implement it during M0-M5.

---

# 19. Leakage rules

The following are prohibited.

## Target-reader leakage

For a held-out reader:

```text
no target utility labels in training
no target utility labels in dev
no target utility labels in feature normalization
no target utility labels in fingerprint construction
no target utility labels in hyperparameter selection
```

## Event leakage

All model splits are event-disjoint.

Never let different cutoffs of the same event cross train/dev/eval boundaries.

## Probe leakage

Probe events come only from foundation_train.

No utility_eval or confirmatory event may enter the fingerprint probe set.

## Gold leakage

Gold rumor labels may be used:

```text
to construct utility supervision
to balance frozen event sampling
to evaluate rumor correctness
```

They may not be input features to utility prediction or fingerprinting.

---

# 20. Feature normalization

All feature scalers must be fitted on the relevant training fold only.

For unseen-reader evaluation, reader-fingerprint normalization statistics must
be fitted using training readers only.

Do not standardize using held-out reader population statistics except where a
feature is intrinsically self-normalized within a reader; such features must be
explicitly documented.

---

# 21. Model-selection discipline

No broad hyperparameter search.

Freeze a small grid during M1:

```text
hidden size: 32
dropout: {0.0, 0.1}
lr: {1e-3, 3e-4}
weight decay: {0, 1e-4}
```

Select using utility_dev only.

After M3 begins, do not expand the grid.

After M4 confirmatory manifest freeze, no hyperparameter change is allowed.

Seeds:

```text
7319
17319
27319
```

Final prediction is seed-averaged.

---

# 22. Training losses

Main multi-task loss:

```text
L = Huber(utility_hat, utility)
  + lambda_sign * CrossEntropy(sign_hat, sign)
```

Freeze:

```text
lambda_sign = 1.0
Huber delta = 0.05
```

Class imbalance:

Use training-fold class weights for sign CE.

Do not derive class weights from held-out reader or eval events.

No reader-specific fine-tuning.

---

# 23. Reader panel deployment rules

All large model deployment/testing happens on SERVER/DGPA.

Never run large-reader tests locally.

If a new reader requires a Transformers version incompatible with the existing
reader overlay, create a new isolated overlay under:

```text
/data/jyz/next/llm/.bcr_env/<reader>/
```

Do not modify DGPA globally.

Every reader identity includes:

```text
model ID
upstream revision
weight hash
tokenizer hash
chat-template hash
effective transformers version
effective tokenizers version
dtype
```

---

# 24. Permanent execution environment contract

## LOCAL

```text
env = pytorch
role = lightweight code implementation and tests
```

Local allowed tasks:

```text
Git
unit tests
compileall
static verifier
small synthetic tests
report generation
```

Large model weights must not be loaded locally.

If local fallback download is required, weights must be downloaded to a
**non-C drive** only.

Never download model weights to Windows C:.

## SERVER

```text
env = DGPA
root = /data/jyz/next/llm/
role = data-scale analysis, model deployment, reader inference, training, formal experiments
```

All new server files/resources belong under:

```text
/data/jyz/next/llm/
```

except historical authoritative resources already located elsewhere.

---

# 25. Server connection contract

All server commands must use the local `next` wrapper:

```text
next/.codex_tmp_ssh_run.cmd
```

All transfers must use:

```text
next/.codex_tmp_scp_run.cmd
```

If the server/tunnel explicitly refuses a connection:

```text
STOP
preserve exact error
notify user
wait for user to restore tunnel
retry only after explicit user confirmation
```

Do not:

```text
change host
change port
change user
switch SSH client
switch tunnel
repeatedly hammer connection
continue server work locally
```

Connection refusal is:

```text
INFRASTRUCTURE_PAUSE
```

not an experiment failure.

---

# 26. Model download contract

Known server resources must be reused.

Do not redownload:

```text
Qwen3-8B
Mistral-7B-Instruct-v0.3
InternLM3-8B-Instruct
```

Historical Qwen3-0.6B must be searched for before any new download.

For a genuinely missing model:

```text
1. SERVER/DGPA download first
2. store under /data/jyz/next/llm/model/
3. verify on server
```

If server download fails because of network/access:

```text
LOCAL/pytorch
-> exact official model
-> NON-C drive
-> next/.codex_tmp_scp_run.cmd
-> /data/jyz/next/llm/model/
-> server-side verification
```

Actual model load/preflight always occurs on SERVER/DGPA.

Do not substitute another model without research approval.

---

# 27. Artifact retention

Repository should contain:

```text
protocol manifests
hashes
small feature summaries
metrics
reports
verifier output
code/tests
```

Heavy model checkpoints and large intermediate feature matrices may remain
server-only if repository policy already excludes them.

Every server-only artifact must be represented by:

```text
absolute server path
bytes
SHA256
schema/version
```

Do not introduce Git LFS or alter artifact retention merely to force large
files into Git unless explicitly approved.

---

# 28. Verifier requirements

BCR verifier must fail closed on:

```text
historical CR-TSER hash drift
wrong dataset role
wrong reader panel
wrong probe manifest
probe/utility event overlap
target-reader label leakage
held-out reader included in scaler fitting
wrong utility threshold
wrong cutoffs
wrong model identity
confirmatory split reuse
unexpected reader
unexpected model substitution
```

Every formal stage must save:

```text
protocol version
baseline commit
input hashes
output hashes
environment
reader identities
stage verdict
```

---

# 29. Stage approval workflow

The execution agent must stop at each checkpoint.

```text
M0 implementation/bootstrap
    ↓ approval

M1 three-reader mechanism pilot
    ↓ approval

M2 six-reader deployment + atomic labels
    ↓ approval

M3 six-reader transfer development
    ↓ approval

M4 fresh confirmatory evaluation
    ↓ approval

M5 secondary PHEME replication
```

The agent must never automatically continue into the next stage after a GO
result.

A NO-GO result must be recorded honestly and execution stopped.

---

# 30. Immediate first execution round — M0

The first delegated round should implement only M0.

Required:

```text
new BCR package / scripts / verifier
historical read-only importer
frozen CR-TSER hash verification
I1 atomic bootstrap extraction
reader-utility geometry diagnostics
72-item foundation_train probe manifest
synthetic tests
LOCAL and SERVER lightweight verification
```

Forbidden in M0:

```text
reader probing
model download
new LLM inference
new utility labels
predictor training
new reader deployment
```

M0 completion criteria:

```text
historical hashes match
Ma-Weibo atomic keys = 2549
PHEME atomic keys = 2107
probe manifest = 36 events/dataset
12 events/cutoff/dataset
no probe event overlaps historical utility_train/dev/eval
all tests pass
BCR verifier issues = 0
```

Commit/push and STOP.

---

# 31. Required M0 report

Report:

```text
commit SHA
LOCAL/pytorch tests
SERVER/DGPA lightweight checks if needed
new files
historical input SHA256
atomic evidence counts
probe manifest counts
event-overlap audit
geometry diagnostic summary
verifier result
confirmation no model inference / labels / training occurred
```

---

# 32. Publication-grade success claim

Only after M4 passes may the project claim:

> A utility-label-free behavioral representation of an unseen LLM reader
> improves prediction of reader-specific helpful/harmful temporal social
> evidence over reader-agnostic evidence models.

Do not claim:

```text
behavioral similarity guarantees intervention transfer
social propagation structure explains utility
all LLMs share a stable utility geometry
```

unless separately supported.

---

# 33. Key references

1. Zhang et al. **LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation.**  
   arXiv:2510.11358, latest revision 2026.  
   https://arxiv.org/abs/2510.11358

2. Zhou. **Preference Is Not Intervention: The Structure and Stability Boundaries of Reader-Specific Evidence Utility.**  
   arXiv:2608.17781, 2026.  
   https://arxiv.org/abs/2608.17781

3. Zhang et al. **Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation.**  
   EMNLP 2025.  
   https://aclanthology.org/2025.emnlp-main.88/

4. Zhang et al. **Beyond Relevance: Utility-Centric Retrieval in the LLM Era.**  
   arXiv:2604.08920, 2026.  
   https://arxiv.org/abs/2604.08920

5. `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.  
   Frozen multilingual NLI feature extractor candidate.  
   https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
