# CR-TSER V2 Reader Protocol Amendment R1 — Implementation + Mistral Preflight

Round: **V2-R1-M0** (reader protocol migration + Mistral preflight)
Baseline: `6da025c` → code commits `0777912`, `4efffbb`
Status: **implementation complete, Mistral preflight 11/12 PASS — one open
finding (`boundaries_ok = false`) returned for research review; STOP**

No formal v2r1 manifest, cache migration, utility label, training run, Stage A/B
or P1–P4 execution was performed.

## 1. What the amendment changed

```text
REMOVED   zai-org/glm-4-9b-chat-hf        (retired from every v2r1 loop)
ADDED     mistralai/Mistral-7B-Instruct-v0.3

new reader keys     ("qwen", "mistral", "internlm")
new LORO rotations  qwen+mistral -> hold internlm
                    qwen+internlm -> hold mistral
                    mistral+internlm -> hold qwen
protocol            DATASET_PROTOCOL_VERSION = "v2"
                    READER_PROTOCOL_VERSION  = "r1"
                    PROTOCOL_VERSION         = "v2r1"
results root        results/cr_tser_v2r1/
```

`glm` keeps its own key and model id (`READER_KEYS_V2` /
`KNOWN_READER_MODEL_IDS`) only so the frozen V2 evidence stays addressable; it
is absent from `READER_KEYS`, from `LORO_ROTATIONS` and from the five v2r1
execution scripts (asserted by test and by the v2r1 verifier). The retired key
was never reused for the replacement reader.

## 2. Code delivered

```text
project/cr_tser/config/pilot_config.py      v2r1 namespace, R1 reader set/LORO,
                                            mistral_model + CRTSER_MISTRAL_MODEL,
                                            protocol version constants
project/cr_tser/readers/mistral_reader.py   new reader (shared prompt + scorer)
project/cr_tser/readers/base_reader.py      registry + KNOWN_READER_KEYS lookup
project/cr_tser/readers/mock_reader.py      history-aware model-id lookup
project/cr_tser/evaluation/unseen_reader.py EXPECTED_HELDOUT_READERS -> R1
scripts/cr_tser_verify_pilot.py             --protocol v2r1 + R1 checks
scripts/cr_tser_migrate_v2_labels.py        new fail-closed cache migration
scripts/cr_tser_p0_audit.py                 reader-scoped sanity + GPU/prompt evidence
scripts/cr_tser_p0_preflight.py, cr_tser_run_pilot.py  reader names
project/cr_tser/tests/test_v2r1_reader_amendment.py    22 new tests
```

Untouched: sequence-scoring mathematics, candidate A/B definition, prompt
semantics, SRC, interventions, utility equation, BiTTE, predictors, losses,
selection, P1–P4 thresholds, split and cutoffs.

The Mistral reader uses `AutoTokenizer` / `AutoModelForCausalLM`,
`trust_remote_code=False`, `torch_dtype=bfloat16`, `device_map="cuda"`,
`thinking=False`, and reuses `candidate_logprobs_hf()` + `ab_scores()` +
`reader_identity()` — no model-specific generation path exists in it.

## 3. LOCAL/pytorch

```text
python -m pytest project/cr_tser/tests -q                     165 passed
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2r1
                                                              issues = 0
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
                                                              issues = 0
python scripts/cr_tser_verify_pilot.py --mode code --protocol v1
                                                              issues = 0
python -m compileall project/cr_tser scripts                  clean
```

The v2r1 suite specifically covers: exact reader keys/model ids/LORO,
GLM absent from execution loops, GLM still addressable as history, Mistral
path/env resolution, shared-scorer reuse, v2/v2r1 namespace separation, the
frozen-V2 manifest pins, and eleven fail-closed migration behaviours
(valid cache accepted, GLM rows refused, duplicates refused, manifest drift
refused, source drift refused, reader-identity drift refused, non-empty target
refused, missing target namespace refused, dry-run writes nothing).

## 4. SERVER/DGPA

```text
workspace      /data/jyz/next/llm/cr_tser_ws
HEAD           4efffbbc8015154743fc73b424d7852d0f89afcf
conda env      DGPA
overlay        PYTHONPATH=/data/jyz/next/llm/.cr_tser_v2p0/tf453
python 3.11.13 / torch 2.7.1+cu128 / transformers 4.53.3 / tokenizers 0.21.4
GPU            NVIDIA GeForce RTX 4090 (shared, ~4 GB held by another user)
wrappers       next/.codex_tmp_ssh_run.cmd, next/.codex_tmp_scp_run.cmd
```

GitHub egress from the server failed (`GnuTLS recv error`, then
`Failed to connect to github.com port 443`), so the second commit was
transferred as a small git bundle over the SSH channel and fetched locally:
no host/port/user/connection change. `DGPA` site-packages were not modified.

Overlay addition required by the new reader (isolated target directory, DGPA
untouched): `protobuf 7.36.1` + `sentencepiece 0.2.2`. Without protobuf the
Mistral SentencePiece tokenizer cannot be constructed (`ImportError`); that
first preflight attempt is preserved with its error in
`.cr_tser_v2p0/r1_preflight_master.log` history.

## 5. Mistral acquisition and identity

```text
acquisition mode   SERVER download via HF mirror (HF_ENDPOINT=hf-mirror.com)
                   huggingface_hub.snapshot_download
official model id  mistralai/Mistral-7B-Instruct-v0.3
resolved revision  c170c708c41dac9275d15a8fff4eca08d52bab71
server path        /data/jyz/next/llm/model/mistral-7b-instruct-v0.3
```

Weight shards — every indexed shard's full sha256 equals the hub LFS hash:

| file | bytes | sha256 vs hub |
|---|---|---|
| model-00001-of-00003.safetensors | 4 949 453 792 | MATCH |
| model-00002-of-00003.safetensors | 4 999 819 336 | MATCH |
| model-00003-of-00003.safetensors | 4 546 807 800 | MATCH |

`model.safetensors.index.json` references exactly those three shards and all
three are present and verified (`index_shards_complete = true`).

Not downloaded (recorded for transparency): `consolidated.safetensors` — a
redundant re-pack of the same weights that upstream ships alongside the shards
— plus `tokenizer.model.v3` and `.gitattributes`. The indexed shard set is the
authoritative checkpoint transformers loads.

```text
config          vocab_size 32768, torch_dtype bfloat16, 32 layers
config sha256   affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99
index sha256    e489ba553b87cde188d921b1a8283c2e0b9d33d635b88147d96ff0fcd6250016
weight_hash     6491d69e717ce45baaffaef7d668041897a10b3315bdc98f6dbf13f660899abe
tokenizer_hash  5c1a8c918530a4d75fb5e19223ef7e88343c71fe1313254b65f3e54576438340
chat_template_hash e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826
reader_identity_hash 6d521c764a7812f75b5a48313c8444d320453b80cf74cc78140e6f31672f6d78
tokenizer class LlamaTokenizerFast, vocab 32768, chat template present
```

## 6. Mistral load + A/B sanity (20 prompts x 2 passes)

```text
loaded                     true
dtype                      bfloat16
device                     cuda
identical_predictions      true
identity_rate              1.0
finite_logprobs            true
no_empty_continuation      true     (A -> 1 token, B -> 1 token)
system_prompt_preserved    true
user_prompt_preserved      true
generated_text_used        false
scoring_mode               teacher_forced_logprob_sum
boundaries_ok              false    <-- open finding, see §7
label_distribution         A 20 / B 0 (fixed synthetic sanity prompts)
```

The official Mistral template *does* keep the system prompt: it renders the
system message inline as `[INST] <system>\n\n<user>[/INST]`, so the shared
`SYSTEM_PROMPT` and `USER_TEMPLATE` text both survive verbatim.

## 7. Open finding — A/B continuation boundary

`boundaries_ok = false` is the only required check that failed. Root cause,
measured with the tokenizer alone (no model, no GPU):

```text
chat prompt ends with            "...number 0.[/INST]"   (no trailing space;
                                  add_generation_prompt adds nothing)
tokenize(prompt)        last id  4        (= "]")
tokenize(prompt + "A")  last id  29509    (= "]A" merged)
tokenize("A")                    1098
=> tokenize(prompt + "A") != tokenize(prompt) + tokenize("A")
```

Adding a space or a newline does not restore the identity (both measured); the
SentencePiece boundary simply falls inside `[/INST]A`. The consequence is that
`sequence_logprob` feeds the model the concatenation `[..., 4, 1098]` while a
natural rendering of the same characters would be `[..., 29509]`. Scoring stays
deterministic and finite (identical predictions, finite log probs), but the
A/B score is taken at a token position Mistral would not itself produce, and
Mistral is the only one of the three readers where this happens.

No workaround was applied: the prompt text, the chat template, the scoring
mathematics and the token budget are all frozen by the amendment. Options for
research review (not implemented, not chosen here):

```text
A. accept the concatenated tokenization and record it as a reader-specific
   tokenization property of the R1 set;
B. change the rendered prompt for the affected reader (would change prompt
   semantics and make the three readers non-isomorphic);
C. align the scored position with the natural tokenization
   (would change sequence-scoring mathematics);
D. select a different replacement reader (explicitly out of scope for this
   round).
```

## 8. GPU memory — the resource blocker is resolved

```text
total                 24083 MiB
free before load      19651 MiB
peak allocated        13854 MiB
peak reserved         13882 MiB
free after scoring     5733 MiB
```

Mistral peaks at **13.9 GiB**, versus the ~19.3 GiB the retired GLM reader
needed before its scoring peak — it ran to completion on the shared card while
another user held ~4 GiB, which is exactly the blocker this amendment targets.
The protocol-level verdict on `boundaries_ok` is still the one in §7.

## 9. v2r1 verifier

```text
SERVER/DGPA  cr_tser_verify_pilot.py --mode code --protocol v2r1
LOCAL/pytorch                                         issues = 0, pending = 0
```

R1 checks include: exact reader keys/model ids/LORO, protocol version
constants, v2r1 namespace separation, GLM absent from the five execution
scripts, GLM retained only as history, Mistral path/env resolution, Mistral
reader reusing the shared scorer, and the six frozen-V2 manifest digests.

## 10. Historical namespaces untouched

`results/cr_tser_v2/` is unmodified — the six pinned manifest digests
(`source.json`, `hashes.json`, `event_split.json` for both datasets) still
match, no GLM row was deleted or renamed, no old cache was overwritten, and a
verifier re-run writes only into the namespace it verified. The v2 and v1 code
verifiers still report `issues = 0` against their own frozen reader sets.

## 11. Evidence files

```text
results/cr_tser_v2r1/reader_amendment/mistral_identity.json
    rev c170c708..., all hub files with local sha256, shard match table,
    weight/tokenizer/config/index hashes
results/cr_tser_v2r1/reader_amendment/mistral_preflight.json
    full sanity record: identity, tokenization example, boundary ids,
    rendered prompt head, GPU memory, required-check table
results/cr_tser_v2r1/reader_amendment/mistral_boundary_diagnosis.txt
    tokenizer-only boundary measurements (chat tail, template, id tables)
```

## 12. Not run

```text
formal v2r1 manifests                NOT RUN
formal Qwen/InternLM cache migration NOT RUN (utility implemented + tested only)
Mistral utility labels               NOT RUN
cr_tser_train_predictors.py          NOT RUN
Stage A / Stage B                    NOT RUN
held-out-reader evaluation           NOT RUN
P1-P4 gates / final pilot report     NOT RUN
```
