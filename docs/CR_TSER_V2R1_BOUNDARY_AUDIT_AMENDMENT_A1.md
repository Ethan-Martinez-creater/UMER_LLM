# CR-TSER V2R1 Boundary Audit Amendment A1

## Decision
Baseline: `27dc0a6cfcf7ebc370a0edf1b0869381ee7ac762`

Reader set remains:
- Qwen/Qwen3-8B
- mistralai/Mistral-7B-Instruct-v0.3
- internlm/internlm3-8b-instruct

Mistral deployment/resource feasibility is accepted. The only open issue is the legacy `boundaries_ok=false`.

This amendment changes only the **v2r1 boundary audit criterion**. It does not change prompt semantics, A/B labels, teacher-forced scoring mathematics, model weights, utility definition, datasets, split, cutoffs, or P1–P4 thresholds.

## 1. Reason
The legacy diagnostic requires:

`tokenize(rendered_prompt + candidate) == tokenize(rendered_prompt) + tokenize(candidate)`

Mistral's SentencePiece tokenizer can merge tokens across the literal `[/INST]` + candidate text boundary when the whole string is re-tokenized. Actual autoregressive inference instead predicts the next token from an already-tokenized prompt prefix.

Historical V1/V2 boundary semantics stay unchanged.

## 2. New v2r1 boundary gate
For each reader and candidate A/B:

1. Compute tokenizer-native prompt IDs with:
`tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)`

2. Compute scorer prompt IDs using the existing text path:
`apply_chat_template(..., tokenize=False)` followed by `tokenize_prompt(...)`.

3. Require exact equality:
`native_prompt_ids == scorer_prompt_ids`

4. Candidate IDs remain:
`tokenizer(candidate, add_special_tokens=False)["input_ids"]`

Require:
- non-empty candidate IDs;
- no unexpected special/control token;
- decoding the candidate IDs recovers the intended label under an explicit tested normalization.

5. Scoring remains unchanged:
`sum log p(candidate_token_k | prompt_ids, previous_candidate_tokens)`

Do not change `sequence_logprob`, `normalize_ab`, A/B definitions, prompt text, chat template, or token budget.

## 3. Legacy diagnostic
Keep the old text-retokenization check as an informational field such as:
`text_retokenization_stable`.

For v2r1, the formal gate becomes:
`autoregressive_boundary_ok`.

Thus Mistral may legitimately record:
- `text_retokenization_stable = false`
- `autoregressive_boundary_ok = true`

only under this amendment.

## 4. Required pass condition
All three readers must satisfy:
- native/scorer prompt IDs identical;
- valid A and B candidate IDs;
- finite logprobs;
- non-empty continuations;
- repeated predictions identical;
- `identity_rate = 1.0`;
- `autoregressive_boundary_ok = true`.

Run 20 sanity prompts × 2 passes per reader.

## 5. Execution environment
LOCAL:
- env = `pytorch`
- code/tests only
- no 7B/8B model loading

SERVER:
- env = `DGPA`
- root = `/data/jyz/next/llm/`
- real-reader preflight only

Server command wrapper:
`next/.codex_tmp_ssh_run.cmd`

Transfer wrapper:
`next/.codex_tmp_scp_run.cmd`

If the server/tunnel explicitly refuses connection:
STOP, preserve the exact error, notify the user, and wait for tunnel restoration. Do not change host/port/user/connection method.

Existing models must be reused:
- Qwen3-8B: `/data/jyz/next/llm/model/qwen3-8b`
- Mistral-7B-Instruct-v0.3: `/data/jyz/next/llm/model/mistral-7b-instruct-v0.3`
- InternLM3-8B-Instruct: `/data/jyz/next/llm/model/internlm3-8b-instruct`

Historical Qwen3-0.6B and retired GLM are not v2r1 readers.

Reader runtime overlay:
`/data/jyz/next/llm/.cr_tser_v2p0/tf453`
with transformers 4.53.3 / tokenizers 0.21.4.

## 6. Allowed code changes
Allowed:
- v2r1 boundary audit/report fields;
- v2r1 verifier;
- tests;
- documentation.

Forbidden:
- scoring mathematics;
- prompt semantics;
- model/dtype changes;
- SRC/interventions/utility;
- BiTTE/predictors/training;
- reader membership/LORO;
- P1–P4 thresholds;
- split/cutoffs.

If scorer prompt IDs differ from tokenizer-native prompt IDs for any reader, STOP and return for research review.

## 7. Tests
At minimum:
- V1/V2 historical boundary semantics unchanged;
- v2r1 uses `autoregressive_boundary_ok`;
- native/scorer prompt ID equality;
- candidate A/B decode validity;
- special-token rejection;
- empty-candidate rejection;
- Mistral `text_retokenization_stable=false` remains visible;
- synthetic prompt-ID mismatch fails closed;
- Qwen/InternLM regression.

LOCAL commands:
```bash
python -m pytest project/cr_tser/tests -q
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2r1
python -m compileall project/cr_tser scripts
```

## 8. Server verification
Run all three reader sanity checks on SERVER/DGPA and record:
- native_prompt_matches_scorer;
- A/B candidate ids and decode;
- autoregressive_boundary_ok;
- text_retokenization_stable;
- finite_logprobs;
- identical_predictions;
- identity_rate.

## 9. Stop rule
If all three readers pass the amended gate:
`V2R1_READER_PREFLIGHT = PASS`

Commit/push and STOP.

Do not yet:
- build formal v2r1 manifests;
- migrate formal Qwen/InternLM caches;
- generate Mistral utility labels;
- train predictors;
- run Stage A/B;
- run P1–P4.

If any reader fails, STOP and preserve evidence for research review.
