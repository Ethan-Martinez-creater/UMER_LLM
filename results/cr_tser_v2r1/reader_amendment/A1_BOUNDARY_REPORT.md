# CR-TSER V2R1 Boundary Audit Amendment A1 — reader preflight report

Baseline commit: `27dc0a6cfcf7ebc370a0edf1b0869381ee7ac762`
Amendment commit: `c5bb07841f064f54125cdd368340316c4b7cd4a0`
Amendment specification: `docs/CR_TSER_V2R1_BOUNDARY_AUDIT_AMENDMENT_A1.md`

**Verdict: `V2R1_READER_PREFLIGHT = PASS`** — all three readers satisfy the
amended gate with 20 sanity prompts × 2 repeated scoring passes each.

## 1. Decision and reason

The reader set is unchanged:

| key | model id | server path |
|---|---|---|
| qwen | `Qwen/Qwen3-8B` | `/data/jyz/next/llm/model/qwen3-8b` |
| mistral | `mistralai/Mistral-7B-Instruct-v0.3` | `/data/jyz/next/llm/model/mistral-7b-instruct-v0.3` |
| internlm | `internlm/internlm3-8b-instruct` | `/data/jyz/next/llm/model/internlm3-8b-instruct` |

Mistral keeps its place; `boundaries_ok=false` is **not** waived. The v2r1
boundary gate is amended to the tokenizer-native autoregressive identity while
the legacy text-retokenization check stays visible as an informational field.
Nothing else changes: the scoring mathematics, the prompt text, the chat
template, the A/B candidate definitions, the model/dtype, the datasets, the
split and the P1–P4 thresholds are untouched.

Why the legacy check was the wrong criterion: it asks whether
`tokenize(prompt + candidate)` equals `tokenize(prompt) + tokenize(candidate)`.
Autoregressive inference never re-tokenizes an already-tokenized prompt prefix —
it predicts the next token from the prompt ids. Mis-Mistral's SentencePiece
tokenizer merges the token straddling the literal `[/INST]` seam, so the
concatenation identity fails even though the scorer's prompt ids are exactly the
ids the tokenizer itself would produce. The amended gate tests that identity
directly.

## 2. Amended gate

For every reader, on the frozen prompt of the first sanity example:

```text
native_prompt_ids  = apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=True)
scorer_prompt_ids  = tokenize_prompt(apply_chat_template(messages,
                                        tokenize=False))
```

Required: `native_prompt_ids == scorer_prompt_ids`, and both A/B candidates have

* non-empty continuation ids;
* no unexpected special/control token;
* a decode that recovers the intended label under the explicit tested
  normalization (`strip + collapse whitespace + upper`).

`autoregressive_boundary_ok = native_prompt_matches_scorer AND
candidate_ids_valid` is the formal v2r1 gate. `text_retokenization_stable` (the
legacy concatenation identity, including each candidate's joined tail) is
recorded in full and never hidden.

## 3. Results

| reader | native prompt ids | scorer prompt ids | `native_prompt_matches_scorer` | A ids / decode | B ids / decode | `autoregressive_boundary_ok` | `text_retokenization_stable` |
|---|---|---|---|---|---|---|---|
| qwen | 70 | 70 | **true** | `[32]` → `"A"` | `[33]` → `"B"` | **true** | true |
| mistral | 65 | 65 | **true** | `[1098]` → `"A"` | `[1133]` → `"B"` | **true** | **false** |
| internlm | 72 | 72 | **true** | `[28005]` → `"A"` | `[28022]` → `"B"` | **true** | true |

Mistral is exactly the case amendment A1 describes. Its rendered prompt ends
with `[/INST]` (last native id `4`), and re-tokenizing `prompt + "A"` yields the
single merged id `29509` in place of the candidate's own `[1098]`. The joined
tail is therefore `[29509] ≠ [1098]`, so the legacy concatenation check fails,
while the amended gate passes because the prompt prefix the scorer conditions on
is byte-identical to the tokenizer-native prefix (both `65` ids, same hash).
The merge is a re-tokenization artifact of the diagnostic, not of the inference
path.

Per-reader sanity, 20 prompts × 2 passes:

| reader | `identical_predictions` | `identity_rate` | `finite_logprobs` | empty continuations | continuation tokens | label distribution |
|---|---|---|---|---|---|---|
| qwen | true | 1.0 | true | none | A=1, B=1 | A=20, B=0 |
| mistral | true | 1.0 | true | none | A=1, B=1 | A=20, B=0 |
| internlm | true | 1.0 | true | none | A=1, B=1 | A=17, B=3 |

First-example teacher-forced scores (log p sums, `A = RUMOR`):

* qwen `A=-0.001170`, `B=-6.751170` → `p_rumor=0.998830`
* mistral `A=-0.222088`, `B=-1.972088` → `p_rumor=0.851953`
* internlm `A=-0.710921`, `B=-0.710921` → tie at `0.5` (the frozen tie rule
  resolves to `A`)

internlm's tie is a property of the deliberately contentless sanity prompt, not
of the scoring path; repeated scoring reproduces it exactly.

All 14 required checks pass for every reader: loaded, bf16, cuda,
`native_prompt_matches_scorer`, `candidate_ids_valid`,
`autoregressive_boundary_ok`, no empty continuation, finite log probs,
`identical_predictions`, `identity_rate = 1.0`, system prompt preserved, user
prompt preserved, generated text not used, teacher-forced scoring.

## 4. Reader identity

| reader | dtype / device | weight hash | tokenizer hash | chat template hash | reader identity hash |
|---|---|---|---|---|---|
| qwen | bfloat16 / cuda | `345a676964219bec144ec6d9d9eaf7c9f53b8e53f73d47b0982531f91007b197` | `26a5805b76938647244b41b841fa742e0270bba814de7e4f71c5d655721da457` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` | `3fc8ff4659f9502077ad4bade9c547fd4f5facf7e20594d1a0498079b927704d` |
| mistral | bfloat16 / cuda | `6491d69e717ce45baaffaef7d668041897a10b3315bdc98f6dbf13f660899abe` | `5c1a8c918530a4d75fb5e19223ef7e88343c71fe1313254b65f3e54576438340` | `e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826` | `6d521c764a7812f75b5a48313c8444d320453b80cf74cc78140e6f31672f6d78` |
| internlm | bfloat16 / cuda | `3ac547bf153173e20c832a3331c9e9098479cf9c8e546992e0171288fb3bdcff` | `c0ed5fe5c9f2d713b8704acf29e75f408aea50f7cdc5fce7fe4dcf57879a6491` | `49aa7cb516942cc5a17d45ed2939828e3b5e4188dc487a3c4b64a570e5b5111f` | `c023219fa6b4af480a9c06e4b2521a95e1d218ae53008960e2d4e9dbb419295d` |

The qwen and internlm hashes match the ones recorded in the frozen V2 reader
audit; the models were reused as deployed, nothing was re-downloaded.

## 5. GPU memory (RTX 4090, 24564 MiB total)

| reader | weight bytes (MiB) | required free (MiB) | free before load | peak allocated | peak reserved | free after |
|---|---|---|---|---|---|---|
| qwen | 15623 | 17671 | 23498 | 15737 | 15856 | 3761 |
| mistral | 13825 | 15873 | 19597 | 13854 | 13890 | 5707 |
| internlm | 16793 | 18841 | 19577 | 16893 | 16980 | 2777 |

No other compute process held the GPU at launch (193 MiB used, 23891 MiB free),
and no reader approached the shared 24 GiB budget: the largest peak is internlm
at 16893 MiB. The retired GLM reader needed ~19.3 GiB and OOMed twice; the R1
reader set removes that blocker entirely.

## 6. Environments

LOCAL (`pytorch`, Python 3.9.18) — code and tests only, no model loading:

* `python -m pytest project/cr_tser/tests -q` → **185 passed**
* `python scripts/cr_tser_verify_pilot.py --mode code --protocol v1|v2|v2r1` →
  `issues = 0, pending = 0` for all three protocols
* `python -m compileall project/cr_tser scripts` → clean

SERVER (`DGPA`, Python 3.11.13): torch `2.7.1+cu128`, transformers `4.53.3`,
tokenizers `0.21.4`, runtime overlay `/data/jyz/next/llm/.cr_tser_v2p0/tf453`
(DGPA site-packages untouched), workspace `/data/jyz/next/llm/cr_tser_ws` at
`c5bb078`. The v2r1 code verifier on the server also reports
`issues = 0, pending = 0`.

## 7. Artifacts

| file | sha256 |
|---|---|
| `a1_boundary_preflight.json` | `1d64957cf1d1256652ac688e2e1f2286aca4a89d7f3625ed4c2b163e3e33eb36` |
| `a1_boundary_preflight.log` | `ec3bece481c98418e478dad6506f4ea9e8f9d72f4e2935cd22b0185c4b84b735` |

## 8. Frozen evidence untouched

* `results/cr_tser_v2/` — the V2 utility-label cache is byte-identical before
  and after the server checkout (maweibo `6525` rows, pheme `5758` rows, same
  sha256), and the frozen V2 manifests still match their pins
  (`v2_history_untouched: pass`). The only server-side V2 changes were the
  regenerable verifier outputs, which were restored to their committed state.
* V1/V2 boundary semantics are unchanged: `boundary_gate_key("v1") ==
  boundary_gate_key("v2") == "boundaries_ok"`, `ab_token_report` still returns
  `all_boundaries_ok`, and the AST digests of `sequence_logprob`,
  `normalize_ab`, the tokenizers and the legacy boundary check are pinned in the
  verifier (`a1_scoring_math_unchanged: pass`).

## 9. Not run

No formal v2r1 manifest, no Qwen/InternLM cache migration, no Mistral utility
label, no predictor training, no Stage A/B and no P1–P4 gate was run. The round
stops at `V2R1_READER_PREFLIGHT = PASS`.
