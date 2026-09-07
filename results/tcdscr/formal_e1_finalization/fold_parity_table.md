# Fold-ID parity audit — old UMER vs TC-DSCR (E1 finalization)

partition_seed=3090, 5 outer folds, 10% inner validation. gold = historical runner functions on the old event ordering/labels; new = TC-DSCR Protocol A on the sorted TC-DSCR registry. PASS = exact match on all three sets and zero cross-set overlap.

| dataset | fold | train exact | val exact | test exact | old-trn ∩ new-tst | old-trn ∩ new-val | old-val ∩ new-tst | gold==sim | PASS |
|---|---|---|---|---:|---:|---:|---|---|
| pheme | 0 | False | False | False | 936 | 364 | 91 | True | False |
| pheme | 1 | False | False | False | 938 | 372 | 105 | True | False |
| pheme | 2 | False | False | False | 947 | 374 | 99 | True | False |
| pheme | 3 | False | False | False | 945 | 358 | 89 | True | False |
| pheme | 4 | False | False | False | 953 | 372 | 107 | True | False |
| maweibo | 0 | False | False | False | 668 | 267 | 72 | True | False |
| maweibo | 1 | False | False | False | 679 | 264 | 66 | True | False |
| maweibo | 2 | False | False | False | 681 | 264 | 77 | True | False |
| maweibo | 3 | False | False | False | 666 | 271 | 78 | True | False |
| maweibo | 4 | False | False | False | 675 | 271 | 68 | True | False |

Overall: UMER_INIT_FOLD_PARITY_FAIL
- folds passing: 0 / 10
