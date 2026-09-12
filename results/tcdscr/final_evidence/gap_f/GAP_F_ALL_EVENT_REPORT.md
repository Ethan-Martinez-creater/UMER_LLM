# Gap F — Dynamic V1 All-Event Held-out Report

Protocol: ALL_EVENT outer test split, frozen per-fold best_config (no test-time search), 30 runs, paired event bootstrap 10000 / seed 4096, fold-stratified.

## PHEME

- Static: 0.84660
- Dynamic: 0.84669
- Delta: +0.000091 CI [-0.000084, +0.000270]
- FlipRate delta: +0.000550 CI [+0.000322, +0.000799]
- **Status: DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED**

| cutoff | Static | Dynamic |
|---|---:|---:|
| 5m | 0.8425 | 0.8425 |
| 15m | 0.8454 | 0.8454 |
| 30m | 0.8478 | 0.8476 |
| 60m | 0.8472 | 0.8475 |
| 180m | 0.8484 | 0.8487 |
| 360m | 0.8481 | 0.8484 |

| fold | Static | Dynamic |
|---|---:|---:|
| 0 | 0.8413 | 0.8414 |
| 1 | 0.8653 | 0.8650 |
| 2 | 0.8284 | 0.8286 |
| 3 | 0.8396 | 0.8396 |
| 4 | 0.8577 | 0.8581 |

## Ma-Weibo

- Static: 0.92443
- Dynamic: 0.92444
- Delta: +0.000012 CI [-0.000131, +0.000155]
- FlipRate delta: -0.000014 CI [-0.000214, +0.000200]
- **Status: DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED**

| cutoff | Static | Dynamic |
|---|---:|---:|
| 5m | 0.8937 | 0.8937 |
| 15m | 0.9177 | 0.9177 |
| 30m | 0.9255 | 0.9256 |
| 60m | 0.9312 | 0.9311 |
| 180m | 0.9369 | 0.9370 |
| 360m | 0.9416 | 0.9417 |

| fold | Static | Dynamic |
|---|---:|---:|
| 0 | 0.9236 | 0.9234 |
| 1 | 0.9278 | 0.9276 |
| 2 | 0.9284 | 0.9285 |
| 3 | 0.9376 | 0.9378 |
| 4 | 0.9047 | 0.9049 |

