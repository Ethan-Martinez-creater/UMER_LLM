# CR-TSER P0 preflight report

## 1. Frozen code commit

- `ed2e19eed813c0869309a8b6a87c4e2c887d9e9e`
- working tree considered clean for tracked files at preflight time

## 2. Candidate Weibo22 sources investigated

| source | identity claimed | attributable to Weibo22 | verdict |
|---|---|---|---|
| KPG official repository — data/Weibo | Weibo22 (CUHK/KPG COVID-19 Weibo propagation dataset, 2087 rumor + 2087 non-rumor source events) | `True` | `ACCEPT_AS_WEIBO22_SOURCE` |
| CUHK RDM fund — Weibo22/KPG project description | the same CUHK Weibo platform COVID-19 dataset (Nov 2019 – Mar 2022, 2087 rumors + 2087 non-rumors, propagation records) | `True` | `DESCRIPTION_ONLY_NO_DATA_PACKAGE` |
| Zenodo 'Weibo-Covid-19' (Song, Yunya; doi:10.5281/zenodo.13787781) | Weibo COVID-19 vaccine discourse corpus | `False` | `REJECT_SOURCE` |
| Ma-Weibo (Ma et al., IJCAI 2016) | Ma-Weibo rumor propagation dataset | `False` | `REJECT_SOURCE` |

## 3. Provenance conclusion

- the local Weibo22 archives are byte-identical to the official KPG release blobs
- `all_accepted_blobs_match = True`
  - `Weibo_label_All.zip`: pinned `9550ee43a895d043e3e142d6d422b965103032d2` vs local `9550ee43a895d043e3e142d6d422b965103032d2` → match=`True`
  - `data.TD_RvNN.vol_5000.zip`: pinned `46f9af6aec86819e172eb1a6a21d7532e030d5f6` vs local `46f9af6aec86819e172eb1a6a21d7532e030d5f6` → match=`True`

## 4. Raw fields found / missing

| plan §4.1 raw field | status |
|---|---|
| event_source_id | `FIELD_PRESENT` |
| binary_rumor_label | `FIELD_PRESENT` |
| source_text | `FIELD_MISSING` |
| source_timestamp | `FIELD_MISSING` |
| reply_repost_text | `FIELD_MISSING` |
| reply_repost_timestamp | `FIELD_MISSING` |
| current_node_id | `FIELD_PRESENT` |
| parent_node_id | `FIELD_PRESENT` |

- timestamp coverage: `0.0`; source-text coverage: `0.0`; reply-text coverage: `0.0`
- parent coverage: `0.9956609512641872`; event count: `4174`; labels: `{'0': 2087, '1': 2087}`
- no timestamp column exists in either released file; parent index and node index are structural only and are never used as time

## 5. Normalized validation result

- normalized export configured: `False`
- verdict: `WEIBO22_TEMPORAL_UNAVAILABLE`
- reason: the released KPG/TD-RvNN files contain no source/reply text and no absolute source or node timestamp; the plan forbids pseudo-time from row or node order (plan §4.1), so no snapshot can be built

## 6. PHEME smoke result

- status: `PHEME_RAW_MISSING`
- raw dir: `(unset)`
- CRTSER_PHEME_RAW is unset or not a directory on this machine

## 7. Three reader load identities

| reader | frozen model id | resolved path | state | identity hash (16) | A/B sanity |
|---|---|---|---|---|---|
| qwen | `Qwen/Qwen3-8B` | (unset) | exists=`False` loaded=`False` | `7dbed70a7c59b8fe` | MODEL_PATH_MISSING |
| glm | `zai-org/glm-4-9b-chat-hf` | (unset) | exists=`False` loaded=`False` | `71531a577a25a012` | MODEL_PATH_MISSING |
| internlm | `internlm/internlm3-8b-instruct` | (unset) | exists=`False` loaded=`False` | `457374eaa8227455` | MODEL_PATH_MISSING |

- frozen reader weights present locally: `False`; HF-cache matches for the frozen readers: `{'qwen': [], 'glm': [], 'internlm': []}`
- torch `2.5.1+cu118`, transformers `5.14.1`, cuda available `True` (1 device(s))

## 8. A/B scoring sanity

- scoring mode: `teacher_forced_logprob_sum` (no generation, no generated confidence)
- `identical_predictions` required: `False`; `boundaries_ok` required: `False`

## 9. Exact P0 verdict

```text
P0 = P0_FAIL
weibo22_temporal = WEIBO22_TEMPORAL_UNAVAILABLE
weibo22_source_of_record = raw_release
```

## 10. Exact blocker

- the released KPG/TD-RvNN files contain no source/reply text and no absolute source or node timestamp; the plan forbids pseudo-time from row or node order (plan §4.1), so no snapshot can be built
- readers ready: `False` (checked=`True`)
- A/B sanity ok: `False`
- PHEME smoke: `PHEME_RAW_MISSING`

## 11. Next action allowed by the frozen plan

- STOP: a P0 prerequisite is unmet (plan §25); do not generate interventions

No formal manifest, utility label, predictor training, Stage-A freeze, held-out evaluation or P1–P4 run was performed in this preflight.
