# Ma-Weibo cap-aware E1 diagnostic (UMER-init predictions)

**UMER_INIT_RESULT_NOT_YET_APPROVED** — fold-parity audit is not PASS; numbers below are descriptive only and must not be used to select the E2 encoder.

Pooled over 5 folds per seed (every test event scored once); 3-seed mean ± std over the pooled scores. uncapped = cap_hit false; capped = cap_hit true.

## cutoff 180m

| seed | subset | n | accuracy | macro_f1 | weighted_f1 | rumor_f1 | rumor_ratio | nonrumor_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | uncapped | 4417 | 0.9414 | 0.9413 | 0.9414 | 0.9424 | 0.5065 | 0.4935 |
| 2000 | capped | 247 | 0.9798 | 0.9760 | 0.9796 | 0.9664 | 0.3077 | 0.6923 |
| 2001 | uncapped | 4417 | 0.9450 | 0.9449 | 0.9450 | 0.9465 | 0.5065 | 0.4935 |
| 2001 | capped | 247 | 0.9798 | 0.9760 | 0.9796 | 0.9664 | 0.3077 | 0.6923 |
| 2002 | uncapped | 4417 | 0.9400 | 0.9400 | 0.9400 | 0.9415 | 0.5065 | 0.4935 |
| 2002 | capped | 247 | 0.9838 | 0.9809 | 0.9837 | 0.9733 | 0.3077 | 0.6923 |

| seed | subset | n | accuracy | macro_f1 | weighted_f1 | rumor_f1 | rumor_ratio | nonrumor_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3-seed mean±std | || uncapped | 4417.0 | 0.9421 ± 0.0026 | 0.9421 ± 0.0026 | 0.9421 ± 0.0026 | 0.9435 ± 0.0027 | 0.5065 ± 0.0000 | 0.4935 ± 0.0000 || capped | 247.0 | 0.9811 ± 0.0023 | 0.9776 ± 0.0028 | 0.9810 ± 0.0024 | 0.9687 ± 0.0040 | 0.3077 ± 0.0000 | 0.6923 ± 0.0000

## cutoff 360m

| seed | subset | n | accuracy | macro_f1 | weighted_f1 | rumor_f1 | rumor_ratio | nonrumor_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | uncapped | 4304 | 0.9431 | 0.9430 | 0.9431 | 0.9446 | 0.5088 | 0.4912 |
| 2000 | capped | 360 | 0.9833 | 0.9813 | 0.9833 | 0.9752 | 0.3417 | 0.6583 |
| 2001 | uncapped | 4304 | 0.9466 | 0.9465 | 0.9465 | 0.9485 | 0.5088 | 0.4912 |
| 2001 | capped | 360 | 0.9806 | 0.9783 | 0.9805 | 0.9712 | 0.3417 | 0.6583 |
| 2002 | uncapped | 4304 | 0.9440 | 0.9439 | 0.9440 | 0.9459 | 0.5088 | 0.4912 |
| 2002 | capped | 360 | 0.9806 | 0.9783 | 0.9805 | 0.9714 | 0.3417 | 0.6583 |

| seed | subset | n | accuracy | macro_f1 | weighted_f1 | rumor_f1 | rumor_ratio | nonrumor_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3-seed mean±std | || uncapped | 4304.0 | 0.9445 ± 0.0018 | 0.9445 ± 0.0018 | 0.9445 ± 0.0018 | 0.9463 ± 0.0020 | 0.5088 ± 0.0000 | 0.4912 ± 0.0000 || capped | 360.0 | 0.9815 ± 0.0016 | 0.9793 ± 0.0017 | 0.9814 ± 0.0016 | 0.9726 ± 0.0023 | 0.3417 ± 0.0000 | 0.6583 ± 0.0000
