# Weibo22 raw-field audit (plan §30)

- **event_count**: 4174
- **label_distribution**: {'0': 2087, '1': 2087}
- **source_text_coverage**: 0.0
- **reply_text_coverage**: 0.0
- **timestamp_coverage**: 0.0
- **parent_coverage**: 0.9956609512641872
- **duplicate_ids**: 0
- **negative_timestamps**: 0
- **child_earlier_than_parent_count**: 0
- **unresolvable_parent_rate**: 0.0
- **events_with_ge1_valid_reply**: 3173
- **events_viable_15m**: 0
- **events_viable_1h**: 0
- **events_viable_6h**: 0
- **verdict**: WEIBO22_TEMPORAL_UNAVAILABLE

| plan §4.1 required field | status |
|---|---|
| event_source_id | `FIELD_PRESENT` |
| binary_rumor_label | `FIELD_PRESENT` |
| source_text | `FIELD_MISSING` |
| source_timestamp | `FIELD_MISSING` |
| reply_repost_text | `FIELD_MISSING` |
| reply_repost_timestamp | `FIELD_MISSING` |
| current_node_id | `FIELD_PRESENT` |
| parent_node_id | `FIELD_PRESENT` |

> the released KPG/TD-RvNN files contain no source/reply text and no absolute source or node timestamp; the plan forbids pseudo-time from row or node order (plan §4.1), so no snapshot can be built
