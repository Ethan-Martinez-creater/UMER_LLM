# BCR-Utility M1-F — Strong-Baseline & Validity Audit Report

```text
protocol           = bcr_v1
baseline commit    = b62010b07f8fa8fbc9d2604b4e8fdf8cbe6a8d12
verified commit    = 6870220a93c9d450e8c65f8bf5fd2b87b0acc628
environment        = SERVER_DGPA
STATUS             = M1F_CLOSE_BCR_UTILITY_METHOD
```

M1-F is a **bounded falsification round**. It reuses frozen M1/M1-E predictions, features and label caches only: no reader was loaded, no utility label was generated, no E0/E1/E2/E3 cache was regenerated and M2 was not entered. The frozen M1 and M1-E verdicts are unchanged.

## 1. Task validity (Task A)

```text
relabelled historical data = False
utility is truth verification = False
claim allowed  = reader-specific rumour-label evidence utility
claim forbidden = fact-checking / truth-verification utility, or any claim that the evidence was shown to be true or false
validity blocked = False
```

utility is defined over the reader's rumour/non-rumour answer agreement, so an evidence-level claim is supported provided it is stated as rumour-label utility and never as truth verification.

Per dataset:

* **maweibo** — the event's rumour / non-rumour annotation carried by the frozen Ma-Weibo source (Ma et al., IJCAI 2016); the label is a rumour annotation, not a verified true/false verdict on the underlying claim.
* **pheme** — the event's rumour / non-rumour annotation carried by the frozen PHEME source (Zubiaga et al., PLOS ONE 2016); rumour != false and non-rumour != true: PHEME annotates whether a claim circulated as an unverified rumour, not whether it is factually true or false.

Cross-dataset boundary: language-only shift = **False**. Ma-Weibo and PHEME differ in platform, language, annotation process and claim distribution; M1-E measured a systematic E3 distribution shift, so the M1-F contrast must not be reported as a language-only shift.

## 2. Strong baselines (Task B)

### maweibo

| variant | kind | qwen | mistral | internlm | mean |
|---|---|---|---|---|---|
| S0_all_neutral | constant | 0.2954 | 0.3062 | 0.2943 | 0.2986 |
| S1_train_prior | constant | 0.2954 | 0.3062 | 0.2943 | 0.2986 |
| S2_source_state | trained | 0.4385 | 0.2850 | 0.2208 | 0.3148 |
| S3_source_state_plus_z | trained | 0.3014 | 0.3056 | 0.2948 | 0.3006 |
| S4_frozen_d5 | reused | 0.2943 | 0.3035 | 0.3239 | 0.3072 |

```text
dev-selected strong comparator = S2_source_state
selection = mean utility_dev Macro-F1 over the three LORO rotations, training readers only; utility_eval was never consulted
```

Dev Macro-F1 means (`utility_dev`, training readers only):

* S0_all_neutral: 0.2887
* S1_train_prior: 0.2887
* S2_source_state: 0.3580
* S3_source_state_plus_z: 0.3572

### pheme

| variant | kind | qwen | mistral | internlm | mean |
|---|---|---|---|---|---|
| S0_all_neutral | constant | 0.2981 | 0.3282 | 0.2574 | 0.2945 |
| S1_train_prior | constant | 0.2981 | 0.3282 | 0.2574 | 0.2945 |
| S2_source_state | trained | 0.2947 | 0.5223 | 0.1335 | 0.3168 |
| S3_source_state_plus_z | trained | 0.2320 | 0.3909 | 0.2219 | 0.2816 |
| S4_frozen_d5 | reused | 0.1801 | 0.3738 | 0.2711 | 0.2750 |

```text
dev-selected strong comparator = S2_source_state
selection = mean utility_dev Macro-F1 over the three LORO rotations, training readers only; utility_eval was never consulted
```

Dev Macro-F1 means (`utility_dev`, training readers only):

* S0_all_neutral: 0.3015
* S1_train_prior: 0.3015
* S2_source_state: 0.3582
* S3_source_state_plus_z: 0.3572

## 3. Primary comparison (Task C)

```text
primary    = S4_frozen_d5
comparator = S2_source_state
mean delta Macro-F1 = -0.0075  95% CI [-0.0956, +0.0799]
positive readers    = 2/3
worst reader delta  = -0.1443
```

Per held-out reader:

| held-out | Δ Macro-F1 | 95% CI |
|---|---|---|
| qwen | -0.1443 | [-0.3441, +0.0593] |
| mistral | +0.0185 | [-0.0082, +0.0466] |
| internlm | +0.1031 | [+0.0108, +0.1986] |

Secondary metric differences (event-clustered bootstrap):

| metric | Δ | 95% CI |
|---|---|---|
| balanced_accuracy | -0.0936 | [-0.2104, +0.0388] |
| active_macro_f1 | -0.1893 | [-0.3865, +0.0659] |
| harmful_auprc | -0.0419 | [-0.2049, +0.0678] |
| helpful_auprc | -0.0434 | [-0.1031, +0.0152] |
| utility_spearman | -0.0112 | [-0.1846, +0.1337] |
| centered_spearman | +0.0024 | [-0.0994, +0.0825] |
| harmful_f1 | -0.1109 | [-0.3576, +0.1126] |
| helpful_f1 | -0.0852 | [-0.2050, +0.0269] |
| neutral_f1 | +0.0328 | [-0.0619, +0.1531] |

## 4. Active HELPFUL/HARMFUL audit (Task D)

| variant | active n | H-vs-H Macro-F1 | balanced acc | HARMFUL AUPRC | HELPFUL AUPRC | NEUTRAL rate on active |
|---|---|---|---|---|---|---|
| S0_all_neutral | 345 | 0.0000 | 0.0000 | 0.3855 | 0.6716 | 1.0000 |
| S1_train_prior | 345 | 0.0000 | 0.0000 | 0.3855 | 0.6716 | 1.0000 |
| S2_source_state | 345 | 0.4118 | 0.3239 | 0.5175 | 0.6439 | 0.4696 |
| S3_source_state_plus_z | 345 | 0.2068 | 0.1377 | 0.4754 | 0.6171 | 0.7536 |
| S4_frozen_d5 | 345 | 0.2225 | 0.1446 | 0.4860 | 0.5724 | 0.7101 |

A NEUTRAL prediction on a gold-active row counts as an error and is kept in every denominator above.

## 5. Within-snapshot evidence discrimination (Task E)

| variant | snapshots used | rows used | mean snapshot ρ | centered pooled ρ | 95% CI | pairwise concordance |
|---|---|---|---|---|---|---|
| S2_source_state | 156/207 | 1472 | nan | -0.0106 | [-0.0904, +0.0741] | 0.0000 |
| S3_source_state_plus_z | 156/207 | 1472 | -0.0076 | 0.0059 | [-0.0632, +0.0756] | 0.5054 |
| S4_frozen_d5 | 156/207 | 1472 | -0.0028 | -0.0083 | [-0.0551, +0.0409] | 0.5038 |

Centered-Spearman increment over S2:

| variant | Δ centered ρ | 95% CI |
|---|---|---|
| S3_source_state_plus_z | +0.0165 | [-0.0964, +0.1324] |
| S4_frozen_d5 | +0.0024 | [-0.0994, +0.0825] |

## 6. Fixed threshold diagnostics (Task F)

```text
official threshold = +-0.05
diagnostic only    = +-0.03, +-0.05, +-0.07
threshold selected = False
```

Diagnostic Macro-F1 at each fixed threshold (primary S4/D5):

| threshold | Macro-F1 | balanced acc | predicted activity |
|---|---|---|---|
| ±0.03 | 0.2934 | 0.3075 | 0.3121 |
| ±0.05 | 0.2921 | 0.3061 | 0.1058 |
| ±0.07 | 0.2948 | 0.3255 | 0.0191 |

```text
gold activity rate            = 0.1882
predicted activity rate       = 0.3061
near-boundary rate of |u_hat| = 0.1718
near-boundary band            = +-0.05 +- 0.01
A/B label swap                = NOT_TESTABLE_WITHOUT_NEW_READER_INFERENCE
```

## 7. PHEME diagnostic (Task G)

```text
PHEME is diagnostic_only = True
PHEME decides the gate   = False
comparator               = S2_source_state
mean delta Macro-F1      = -0.0419 95% CI [-0.0787, +0.0355]
```

PHEME is a diagnostic only: it cannot decide the primary gate and cannot rescue a failed Ma-Weibo result.

## 8. M1-F decision

```text
mean_delta_gte_0.02                                        = FAIL
ci_low_gt_0                                                = FAIL
positive_readers_gte_2                                     = PASS
active_macro_f1_improves                                   = FAIL
harmful_auprc_not_decreased                                = FAIL
within_snapshot_centered_spearman_improves_over_S2         = PASS
gain_not_only_neutral                                      = FAIL
VERDICT = M1F_CLOSE_BCR_UTILITY_METHOD
```

A CONTINUE outcome permits only a new research-review round for fresh-event / new-reader confirmation; it never authorizes M2.

## 9. Immutability

```text
frozen input groups pinned = 10 (45 artifacts)
M1 verdict unchanged        = True
M1-E verdict unchanged      = True
M2 entered                  = False
results/cr_tser{, _v2, _v2r1} = no tracked changes
new reader inference        = none
new utility labels          = none
regenerated E0/E1/E2/E3     = none
new model deployment        = none (Qwen3-0.6B / Phi / Gemma absent)
```
