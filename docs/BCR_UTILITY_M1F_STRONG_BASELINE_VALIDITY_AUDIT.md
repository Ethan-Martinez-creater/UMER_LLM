# BCR-Utility M1-F Execution Plan
## Strong-Baseline & Validity Audit

Baseline:
- HEAD: `b62010b07f8fa8fbc9d2604b4e8fdf8cbe6a8d12`
- protocol: `bcr_v1`
- M1: `M1_CONDITIONAL_GO`
- M1-E: `COMPLETED`
- M2: `NOT AUTHORIZED`

Purpose: one bounded falsification round using only frozen predictions/features/caches. No new reader inference, no new utility labels, no reader-panel expansion.

## 1. Permanent environment

LOCAL:
- conda env `pytorch`
- implementation, tests, static verifier, reports only
- do not load 7B/8B readers

Known local repo:
`E:\Graduate_work_folder\Graduate_Project_Worksapace\UMER`
Verify with `git rev-parse --show-toplevel`.

SERVER:
- conda env `DGPA`
- root `/data/jyz/next/llm/`
- repo `/data/jyz/next/llm/cr_tser_ws`

Server commands only:
`next/.codex_tmp_ssh_run.cmd`

Transfers only:
`next/.codex_tmp_scp_run.cmd`

Explicit server/tunnel refusal => STOP, preserve exact error, notify user, wait for tunnel restoration. Do not change host/port/user/connection method.

## 2. Existing models

Do not redownload or reload readers:
- Qwen3-8B: `/data/jyz/next/llm/model/qwen3-8b`
- Mistral-7B-Instruct-v0.3: `/data/jyz/next/llm/model/mistral-7b-instruct-v0.3`
- InternLM3-8B-Instruct: `/data/jyz/next/llm/model/internlm3-8b-instruct`

Historical Qwen3-0.6B and retired GLM remain untouched.

Frozen NLI model:
`/data/jyz/next/llm/model/mdeberta-v3-base-mnli-xnli`

M1-F uses only existing M1/M1-E caches and predictions.

## 3. Immutability

Read-only:
- `results/cr_tser/`
- `results/cr_tser_v2/`
- `results/cr_tser_v2r1/`
- existing M0/M1/M1-E artifacts

All new output:
`results/bcr_utility_v1/m1f/`

Before computation pin SHA256 for:
- `M1_VERDICT.json`
- `M1E_VERDICT.json`
- frozen M1 predictions
- frozen D5 predictions
- E0/E1/E2/E3 caches
- historical utility caches
- probe/fingerprint artifacts

Any drift => STOP.

## 4. Task-validity audit

Before performance evaluation, document for each dataset:
- actual dataset label semantics;
- frozen reader prompt target;
- what utility therefore means.

For PHEME explicitly preserve:
`rumour/non-rumour != false/true`.

Do not relabel historical examples. Do not describe Ma-Weibo/PHEME differences as language-only shift.

Outputs:
- `TASK_VALIDITY.md`
- `task_validity.json`

## 5. Frozen evaluation protocol

Primary: Ma-Weibo.
Secondary: PHEME diagnostic only.

Keep:
- utility_train=50
- utility_dev=15
- utility_eval=25
- seed=7319
- three existing LORO rotations

Held-out-reader labels must never enter training, selection, scaler fitting, priors or threshold selection.

## 6. Strong baselines

S0 — all NEUTRAL:
- predict NEUTRAL for every row
- no fitting

S1 — training class prior:
- estimate class distribution using training-reader `utility_train` only
- predict majority class
- no held-out labels

S2 — source-state only:
- `source_only_margin`
- `source_only_entropy`
- `source_nll`
- cutoff
- same frozen M1 grid/seeds

S3 — source-state + simple evidence:
- S2 + E0 + E1 + E2
- no evidence-familiarity C block

S4 — frozen D5:
- E0 + E1 + E2
- source-state S
- evidence-familiarity C
- no behavioral fingerprint
- reuse frozen D5 predictions for existing eval; do not redefine model

Strong simple comparator:
- choose the best of S0-S3 using mean `utility_dev` Macro-F1 on training readers only
- never choose it using utility_eval

Primary comparison:
`S4/D5 vs dev-selected strong simple comparator`.

## 7. Metrics

Overall:
- HELPFUL/NEUTRAL/HARMFUL Macro-F1
- per-class F1
- balanced accuracy
- continuous utility Spearman
- MAE
- HARMFUL AUPRC
- HELPFUL AUPRC

Statistics:
- paired event-level bootstrap
- 10,000 iterations
- seed 7319
- never bootstrap atomic rows independently

## 8. Active signed audit

Gold-active rows:
`HELPFUL or HARMFUL`.

Report:
- HELPFUL-vs-HARMFUL Macro-F1
- balanced accuracy
- HARMFUL AUPRC
- HELPFUL AUPRC
- active prevalence
- NEUTRAL prediction rate on active rows

If the model predicts NEUTRAL for an active example, count it as an error. Do not drop it.

This audit tests whether gains extend beyond NEUTRAL recognition.

## 9. Within-snapshot evidence discrimination

Source-state features are constant inside `(event, cutoff, reader)`.

For each snapshot with >=3 atomic evidence rows and non-zero utility variance compute:
- within-snapshot Spearman(predicted utility, true utility)
- pairwise evidence-order concordance

Also center within snapshot:
- `u_centered = u - mean_snapshot(u)`
- `pred_centered = pred - mean_snapshot(pred)`

Report pooled centered Spearman with event-level bootstrap.

Compare S2, S3 and S4/D5.

Evidence-level utility claims require positive incremental within-snapshot discrimination beyond S2.

## 10. Fixed robustness diagnostics

Official label threshold remains `±0.05`.

Diagnostic only:
- `±0.03`
- `±0.05`
- `±0.07`

Also report:
- continuous utility metrics
- activity rate
- conditional HELPFUL/HARMFUL sign
- frequency near the ±0.05 boundary

Do not choose a new threshold.

If A/B label-swap cannot be reconstructed from frozen caches without new reader inference, record:
`NOT_TESTABLE_WITHOUT_NEW_READER_INFERENCE`
and do not run new reader inference.

## 11. PHEME

Reuse existing M1/M1-E artifacts only.

PHEME remains:
`diagnostic_only = true`

It cannot rescue a failed Ma-Weibo result.

Do not regenerate E3 or rerun reader inference.

## 12. M1-F research decision

### `M1F_CONTINUE_TO_CONFIRMATION_REVIEW`

Require all on Ma-Weibo:
1. S4/D5 vs dev-selected strong simple comparator: mean ΔMacro-F1 >= +0.02
2. paired event-bootstrap 95% CI lower bound > 0
3. >=2/3 held-out readers have positive ΔMacro-F1
4. active HELPFUL/HARMFUL Macro-F1 improves
5. HARMFUL AUPRC does not decrease
6. within-snapshot centered Spearman improves over S2
7. gain is not explained only by predicting NEUTRAL

This outcome does NOT authorize M2. It only permits a new research-review round for fresh-event/new-reader confirmation.

### `M1F_CLOSE_BCR_UTILITY_METHOD`

Use if D5 fails strong simple baselines, active signed discrimination, or within-snapshot evidence discrimination, or the gain is primarily a NEUTRAL/class-prior effect.

Do not rescue with new readers, new heads, wider tuning or threshold changes.

### `M1F_VALIDITY_BLOCKED`

Use if task semantics do not support the intended paper claim.

## 13. Forbidden actions

Do not:
- run Qwen/Mistral/InternLM inference;
- deploy Qwen3-0.6B/Phi/Gemma;
- generate utility labels;
- regenerate E0/E1/E2/E3;
- change probe/fingerprint;
- change utility threshold;
- change reader panel or split;
- expand hyperparameter grid;
- tune on utility_eval;
- enter M2/M3/M4/M5.

No new model architecture.

## 14. Tests/verifier

LOCAL:
```bash
python -m pytest project/bcr_utility/tests -q
python -m pytest project/cr_tser/tests -q
python -m compileall project/bcr_utility scripts/bcr_*.py
```

Add verifier mode `m1f`.

Fail closed on:
- M1/M1-E drift
- CR-TSER historical drift
- unexpected reader inference artifacts
- new utility labels
- regenerated E3
- eval labels used for baseline selection
- held-out-reader leakage
- threshold drift
- PHEME deciding primary result
- M2 artifacts

SERVER authoritative verifier must end:
`issues=0, pending=0`.

## 15. Outputs

`results/bcr_utility_v1/m1f/`
- `evidence_pins.json`
- `TASK_VALIDITY.md`
- `task_validity.json`
- `baselines/`
- `active_signed_audit.json`
- `within_snapshot_audit.json`
- `robustness_diagnostic.json`
- `pheme_diagnostic.json`
- `M1F_VERDICT.json`
- `M1F_REPORT.md`

## 16. Stop rule

Valid outcomes:
- `M1F_CONTINUE_TO_CONFIRMATION_REVIEW`
- `M1F_CLOSE_BCR_UTILITY_METHOD`
- `M1F_VALIDITY_BLOCKED`
- `IMPLEMENTATION_BLOCKED`
- `INFRASTRUCTURE_PAUSE`

Every outcome:
`commit -> push -> STOP`.

Do not automatically start another phase.

## 17. Final report

Report only:
- commit SHA
- LOCAL tests
- SERVER verifier
- task-validity conclusion
- S0/S1/S2/S3/S4 Macro-F1 per held-out reader
- dev-selected strong comparator
- S4 delta / 95% CI / positive readers / worst reader
- active HELPFUL/HARMFUL metrics
- HARMFUL AUPRC
- within-snapshot centered Spearman and pairwise concordance
- fixed ±0.03/±0.05/±0.07 diagnostics
- PHEME diagnostic
- final M1-F verdict
- immutability confirmation
- confirmation of no reader inference/new labels/new model deployment/M2+
