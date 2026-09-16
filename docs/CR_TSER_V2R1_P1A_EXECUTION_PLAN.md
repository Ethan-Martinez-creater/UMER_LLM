# CR-TSER V2R1-P1A Execution Plan

## Approved baseline
- HEAD: `5295d5838be3f78daca53a34e9d6a9c661a45e29`
- `V2R1_READER_PREFLIGHT = PASS`
- Protocol: `v2r1`
- Primary: Ma-Weibo
- Secondary: PHEME
- Readers: Qwen3-8B / Mistral-7B-Instruct-v0.3 / InternLM3-8B-Instruct

This round only:
1. Freeze formal v2r1 manifests.
2. Verify V2 -> v2r1 manifest/source equivalence.
3. Fail-closed migrate existing Qwen/InternLM labels.
4. Generate only Mistral labels for Ma-Weibo and PHEME.
5. Audit final 3-reader caches.
6. Commit/push and STOP.

Do not run predictor training, Stage A/B, held-out evaluation, P1-P4, or final Pilot.

## Permanent execution contract

### LOCAL
- env: `pytorch`
- lightweight tests only
- no 7B/8B model loading

### SERVER
- env: `DGPA`
- root: `/data/jyz/next/llm/`
- formal manifests/cache migration/reader inference/experiments

Reader processes use:
`/data/jyz/next/llm/.cr_tser_v2p0/tf453`
with Transformers 4.53.3 / tokenizers 0.21.4.

Server commands only through:
`next/.codex_tmp_ssh_run.cmd`

Transfers only through:
`next/.codex_tmp_scp_run.cmd`

If server/tunnel explicitly refuses connection:
STOP, preserve the exact error, notify the user, wait for tunnel restoration, then retry only after confirmation. Do not change host/port/user/connection method.

## Existing server models
Do not redownload:
- Qwen3-8B: `/data/jyz/next/llm/model/qwen3-8b`
- historical authoritative Qwen3-8B: `/data/jyz/next/model/qwen3-8b`
- Mistral-7B-Instruct-v0.3: `/data/jyz/next/llm/model/mistral-7b-instruct-v0.3`
- InternLM3-8B-Instruct: `/data/jyz/next/llm/model/internlm3-8b-instruct`

Historical Qwen3-0.6B exists on the server but is not a CR-TSER reader.
Retired GLM remains historical only.

If a future new model is genuinely required: server download first; if that fails, LOCAL/pytorch may download the exact official model to a NON-C drive, then transfer with `.codex_tmp_scp_run.cmd`. Never download model weights to C:.

## Historical artifact contract
`results/cr_tser_v2/` is immutable historical evidence.

Do not rewrite V2 manifests, delete/rename GLM rows, overwrite V2 caches, or alter V2 hashes.

All new formal artifacts go to:
`results/cr_tser_v2r1/`

## LOCAL baseline verification
Run:
```bash
git status
git rev-parse HEAD
python -m pytest project/cr_tser/tests -q
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2r1
python -m compileall project/cr_tser scripts
```

Expected reviewed baseline:
- 185 tests pass
- v2r1 verifier issues=0
- compileall clean

## SERVER source identity
Use HEAD `5295d5838be3f78daca53a34e9d6a9c661a45e29`.

Ma-Weibo must reproduce:
`b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d`

Any source drift => STOP.

Record pre-run hashes of:
- `results/cr_tser_v2/manifests/`
- `results/cr_tser_v2/utility_labels/`

They must remain unchanged.

## Freeze formal v2r1 manifests
Set `CRTSER_OUT_ROOT=results/cr_tser_v2r1` and run:
```bash
python scripts/cr_tser_build_manifests.py --dataset maweibo
python scripts/cr_tser_build_manifests.py --dataset pheme
```

No `--smoke`.

Required Ma-Weibo split remains 80/50/15/25, seed 7319, event-disjoint.

Verify:
- source fingerprint unchanged
- event split identical to V2
- snapshot manifest identical to V2
- intervention manifest identical to V2
- cutoffs only 15/60/360
- no future leakage
- no 1021 cap
- reader set exactly qwen/mistral/internlm
- GLM absent from v2r1 reader records

Record hashes. Once the first v2r1 label row exists, manifests are immutable.

## Dry-run migration first
Source root: `results/cr_tser_v2`
Target root: `results/cr_tser_v2r1`

Run:
```bash
python scripts/cr_tser_migrate_v2_labels.py --dataset maweibo --source-root results/cr_tser_v2 --target-root results/cr_tser_v2r1
python scripts/cr_tser_migrate_v2_labels.py --dataset pheme --source-root results/cr_tser_v2 --target-root results/cr_tser_v2r1
```

Require:
- only qwen/internlm selected
- GLM excluded
- source/manifests/reader identity match
- duplicate keys=0
- target cache empty

Expected if historical cache unchanged:
- Ma-Weibo: qwen 3202 + internlm 3202 = 6404 migrated
- PHEME: qwen 2879 + internlm 2879 = 5758 migrated

Unexpected counts => STOP.

## Apply migration
Only after both dry-runs pass:
```bash
python scripts/cr_tser_migrate_v2_labels.py --dataset maweibo --source-root results/cr_tser_v2 --target-root results/cr_tser_v2r1 --apply
python scripts/cr_tser_migrate_v2_labels.py --dataset pheme --source-root results/cr_tser_v2 --target-root results/cr_tser_v2r1 --apply
```

Require:
- GLM rows in target = 0
- unknown readers = 0
- duplicate keys = 0
- identity fields unchanged

Do not manually edit rows.

## Migration reuse verification
Run the ordinary label generator for Qwen and InternLM against the migrated v2r1 caches.

Expected:
- 0 new Qwen rows
- 0 new InternLM rows
- all expected rows reused
- 0 `CacheIdentityMismatch`

If not, STOP before Mistral generation.

## Generate formal Mistral labels
Only on SERVER/DGPA:
```bash
python scripts/cr_tser_generate_labels.py --dataset maweibo --reader mistral
python scripts/cr_tser_generate_labels.py --dataset pheme --reader mistral
```

No `--smoke` or `--limit-snapshots`.

Expected:
- maweibo/mistral = 3202 rows
- pheme/mistral = 2879 rows

Any identity/source/manifest mismatch => STOP.

## Final cache audit
Expected totals:
- Ma-Weibo: 3202 x 3 = 9606
- PHEME: 2879 x 3 = 8637

Audit:
- reader set exactly qwen/mistral/internlm
- GLM rows=0
- duplicates=0
- missing fields=0
- NaN/Inf=0
- cutoffs only 15/60/360
- source/manifests/reader identities correct
- prompt/context hashes complete
- no smoke contamination

Record per dataset/reader:
row count, bytes, SHA256, cutoff/intervention/sign distributions.

Do not tune thresholds from these distributions.

## Verification
Run:
```bash
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2r1
python scripts/cr_tser_verify_pilot.py --mode pilot --protocol v2r1
```

Downstream predictor/P1-P4 artifacts may remain pending.

Confirm historical V2 hashes are unchanged.

## Stop rule
After manifests + migration + Mistral labels + cache audit + verifier complete:
commit, push, STOP.

Do not run:
- `cr_tser_train_predictors.py`
- Stage A/B
- held-out evaluation
- P1-P4
- final Pilot report

If a genuine code/runtime defect appears: STOP, preserve evidence, do not patch and continue automatically.

## Final report
Report:
- commit SHA
- LOCAL/pytorch checks
- SERVER/DGPA workspace
- connection wrappers
- v2r1 manifest hashes
- migration dry-run/apply results
- migrated Qwen/InternLM row counts
- Mistral row counts
- final cache totals + SHA256
- code/pilot verifier status
- V2 immutability confirmation
- interruptions/resumes
- confirmation training/Stage A/B/P1-P4 NOT RUN
