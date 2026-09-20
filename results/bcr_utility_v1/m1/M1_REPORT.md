# BCR-Utility Phase M1 — Three-Reader Mechanism Pilot Report

```text
protocol        = bcr_v1
approved base   = 6ac2c29c078ff889deec40c03bb63ab0034629af (M0 APPROVED)
code commits    = d92869d (M1 code) → 3685dcf → cae3741 → aba8d6d (probe fixes)
                  → 86a168e (M1-D code) → 13f72dc (E3 validation fix)
final verdict   = M1_CONDITIONAL_GO
```

M1-A/B/C/D executed exactly in the approved order: ZERO-TOUCH first; the
LIGHT-TOUCH fallback (E3/B5) ran **only** after the recorded ZERO-TOUCH
failure. Nothing entered M2.

## 1. Environments

```text
LOCAL  env=pytorch (Python 3.9.18): implementation, tests, static verify
SERVER env=DGPA root=/data/jyz/next/llm workspace=/data/jyz/next/llm/cr_tser_ws
       GPU=NVIDIA RTX 4090 (24 GB), torch 2.7.1+cu128, transformers 4.53.3 (tf453 overlay)
connection: commands only via next/.codex_tmp_ssh_run.cmd;
            transfers only via next/.codex_tmp_scp_run.cmd (no tunnel errors)
```

LOCAL/pytorch tests before server work and after all code changes:

```text
project/bcr_utility/tests : 194 passed (117 M0 + 77 M1)
project/cr_tser/tests     : 219 passed (regression)
compileall                : clean
python scripts/bcr_verify.py --mode m0 : issues=0 (LOCAL & SERVER)
python scripts/bcr_verify.py --mode m1 : issues=0 (SERVER, 0 pending)
                                       : issues=0 (LOCAL, 1 pending = server-only label caches)
```

## 2. Reader identities (stable across both datasets, M1-A audited)

| reader | model_id | dtype | reader_identity_hash |
|---|---|---|---|
| qwen | Qwen/Qwen3-8B | bfloat16 | `3fc8ff4659f9502077ad4bade9c547fd4f5facf7e20594d1a0498079b927704d` |
| mistral | mistralai/Mistral-7B-Instruct-v0.3 | bfloat16 | `6d521c764a7812f75b5a…` |
| internlm | internlm/internlm3-8b-instruct | bfloat16 | `c023219fa6b4af480a9c06e4b2521a95e1d218ae53008960e2d4e9dbb419295d` |

No reader was re-downloaded; no panel expansion; no Phi/Gemma.

## 3. Frozen NLI extractor

```text
model      = MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (exact, no substitution)
server     = /data/jyz/next/llm/model/mdeberta-v3-base-mnli-xnli (SERVER download)
id2label   = {0: entailment, 1: neutral, 2: contradiction} (verified by load)
weight_hash(model_weight_hash) = aec70edcce71ae71978f…
pytorch_model.bin = 557692715 B sha256 345f880b8390c64336cb9fd2907544fc02418bc4e97a74d5952b26b82bfb6f74
config.json       = 1066 B      sha256 4e4430c95100d613df80fa01276f231931e1b535557cf62fbb3cc50323e50cca
tokenizer.json    = 16331396 B  sha256 3aca3ce69a0a35aeb144a52c4f1d41c4246b8785f8f398315cc8fb6b24057810
device = cuda
```

## 4. M1-A probes and fingerprints

Frozen probe manifest reused byte-identically
(`probe_manifest.json` sha256 `48fb3ce44e27330667afedd1c6db1f0aa9a91d72c2cd04c956f7b414ac6c249c`, never rebuilt).

```text
probe coverage = 864/864 rows = 3 readers x 72 items x 4 contexts
per reader     = 288 (qwen 288 / mistral 288 / internlm 288)
response audit = ok (no duplicates, no missing, no NaN/Inf,
                     no utility/gold/label fields, exact model ids)
fingerprints sha256 = 2a5d7cdd1c532c2b34979edac90b7868b3c20c9d53e07076f4e03579f9c4afc5
raw_vectors.json    = 200280 B sha256 1eac16341b5647d51c00a80691ea303d…
```

## 5. Feature caches (rows == frozen atomic keys; E2/E3 == 3x keys)

| cache | maweibo rows | sha256 (first 32) | pheme rows | sha256 (first 32) |
|---|---|---|---|---|
| E0 | 2549 | `f462533dbe4755c8d4c31e53d308b800` | 2107 | `943fce09697d0689a690ecaae36741d2` |
| E1 | 2549 | `8663bc9acf9240ea5f33c61d21647941` | 2107 | `7f99757438e0fe45750c25fcc1364fa5` |
| E2 | 7647 | `c12b6a6764661a4bbe4b7b4068585a8f` | 6321 | `dd6818eaf8a6aa4c10aacb7909db7c92` |
| E3 (M1-D) | 7647 | `8cf80e634b94707b1f85134430e3ba64` | 6321 | `4491ff3692669513f3d724eaae927fd9` |

## 6. M1-C ZERO-TOUCH LORO — Ma-Weibo (primary gate; B4 vs B0)

| held-out | B0 f1 | B3 f1 | B4 f1 | Δ(B4-B0) |
|---|---|---|---|---|
| qwen | 0.2288 | 0.3956 | 0.2400 | +0.0112 |
| mistral | 0.2626 | 0.3959 | 0.1649 | **−0.0978** |
| internlm | 0.2173 | 0.3959 | 0.2924 | +0.0751 |

```text
mean Δ(B4-B0) = −0.0038   95% CI = [−0.0360, +0.0255] (10000 event bootstraps, seed 7319)
positive readers = 2/3    worst reader = −0.0978
gate: FAIL (mean < +0.03, CI covers 0, worst < −0.05)
B2 in-domain diagnostic f1 = 0.2654 (never an unseen-reader comparator)
```

ZERO-TOUCH secondary support: mean ΔSpearman = −0.0092, mean ΔHARMFUL-AUPRC
= +0.0063 (secondary cannot rescue a failed primary gate).

## 7. M1-D LIGHT-TOUCH fallback — Ma-Weibo (B5 vs B0, identical protocol + E3)

E3 features (source-only margin/entropy, source NLL, evidence NLL/token,
conditional evidence NLL, NLL gap) were generated **after** the recorded
ZERO-TOUCH failure only. Same rotations, splits, grid, seeds, metrics,
bootstrap and thresholds as M1-C.

| held-out | B0 f1 | B3 f1 | B5 f1 | Δ(B5-B0) |
|---|---|---|---|---|
| qwen | 0.2288 | 0.3956 | 0.3535 | +0.1247 |
| mistral | 0.2626 | 0.3959 | 0.3095 | +0.0469 |
| internlm | 0.2173 | 0.3959 | 0.2303 | +0.0131 |

```text
mean Δ(B5-B0) = +0.0616   95% CI = [+0.0068, +0.1084]
positive readers = 3/3    worst reader = +0.0131
gate checks: mean≥+0.03 ✓  CI_low>0 ✓  ≥2/3 positive ✓  worst≥−0.05 ✓
```

LIGHT-TOUCH secondary support: mean ΔSpearman = +0.0151 (> 0),
mean ΔHARMFUL-AUPRC = +0.0125 (> −0.02).

Disagreement-focused subset (training-reader signs only), Ma-Weibo f1:

| held-out | n | B0 | B4 (ZT) | B5 (LT) |
|---|---|---|---|---|
| qwen | 174/611 | 0.2413 | 0.2829 | 0.3176 |
| mistral | 188/611 | 0.2323 | 0.1252 | 0.3005 |
| internlm | 197/611 | 0.2357 | 0.3063 | 0.2529 |

## 8. PHEME (diagnostic_only = true, decides_M1_gate = false)

```text
ZERO-TOUCH  mean Δ(B4-B0) = +0.0312  CI=[+0.0033,+0.0622]  pos=2/3 worst=−0.0011
LIGHT-TOUCH mean Δ(B5-B0) = −0.1063  CI=[−0.1367,−0.0759]  pos=1/3 worst=−0.1872
```

PHEME ran the identical pipeline and is reported for diagnosis only; it was
never used to rescue or decide the Ma-Weibo gate. The dataset contrast
(ZERO-TOUCH positive on PHEME, LIGHT-TOUCH strongly negative) is itself a
diagnostic finding for M2 planning.

## 9. Final verdict

```text
M1-C ZERO-TOUCH  : FAIL
M1-D LIGHT-TOUCH : PASS (all four primary conditions)
M1 VERDICT       = M1_CONDITIONAL_GO
```

Artifacts (all under `results/bcr_utility_v1/m1/`):

```text
evaluation.json               1720681 B sha256 3d7461f70e4d19211da66fc8dc90fafd…
evaluation_light_touch.json   1717715 B sha256 2535e3dfc55b2e38bc446f3b91226426…
predictions.jsonl             6414609 B sha256 5669a21f314b2fd1b9bfc43e3f5cb847…
predictions_light_touch.jsonl 6422844 B sha256 d0c6f9b9671a9d8fd92861ea444882af…
gate.json / gate_light_touch.json / M1_VERDICT.json / environment.json
verifier: results/bcr_utility_v1/verifier/bcr_verify_m1_server.json (issues=0, pending=0)
```

## 10. Immutability and boundary confirmations

```text
historical results/cr_tser{,_v2,_v2r1}/ : no tracked changes (verifier m1, SERVER 0 issues)
historical label caches                 : re-hashed on SERVER, unchanged
frozen probe manifest                   : reused, never rebuilt (hash pinned)
utility threshold 0.05 / cutoffs / splits / panel / grid / seeds / gate : unchanged
new utility labels generated            : none
reader panel expansion / Qwen3-0.6B     : none
Phi/Gemma deployed                      : none
M2–M5 / confirmatory experiments        : none executed
```
