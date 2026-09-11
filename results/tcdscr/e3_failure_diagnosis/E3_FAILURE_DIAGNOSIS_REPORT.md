# TC-DSCR E3 Failure Diagnosis

## Overall Finding
MIXED_CAUSES

Evidence-ordered causes:
- DYNAMIC_RANKING_RARELY_CHANGES (pheme, signal=0.0392)
- DYNAMIC_RANKING_RARELY_CHANGES (maweibo, signal=0.0190)
- SELECTION_CHANGES_BUT_PROXY_INSENSITIVE (maweibo, signal=0.0043)

## Bootstrap Fix
- previous bug: with-replacement resample converted into a unique-event dict (multiplicity lost) -> CIs invalid
- corrected implementation: sampled event list keeps every occurrence; 10,000 iterations, seed 3090, paired

- PHEME Macro-F1 CI: [-0.00010, +0.00032]
- PHEME FlipRate CI: [+0.00040, +0.00101]
- Ma-Weibo Macro-F1 CI: [-0.00014, +0.00016]
- Ma-Weibo FlipRate CI: [-0.00024, +0.00021]

## 1. Static vs Dynamic Selection Change
- PHEME: exact-match rate 0.9052, mean Jaccard 0.9691, replacement rate 0.0948
- Ma-Weibo: exact-match rate 0.8852, mean Jaccard 0.9753, replacement rate 0.1148

## 2. Ranking Change
- PHEME: Spearman mean 0.988164, top1 change 0.0253, top5 set change 0.0780
- Ma-Weibo: Spearman mean 0.998364, top1 change 0.0140, top5 set change 0.0513

## 3. Budget Masking
- PHEME: ranking-changed-but-selection-same 0.1990, budget utilization 0.6511
- Ma-Weibo: ranking-changed-but-selection-same 0.4865, budget utilization 0.8833

## 4. Prediction Impact
- PHEME: change-given-selection-change 0.0071, wrong->correct 36, correct->wrong 27, net gain 9
- Ma-Weibo: change-given-selection-change 0.0043, wrong->correct 20, correct->wrong 19, net gain 1

## 5. Score Scale
- PHEME: u std 6.8016, bonus std 0.2666, bonus/utility std ratio 0.0392, mean|bonus|/mean|u| 0.0535
- Ma-Weibo: u std 11.6088, bonus std 0.2206, bonus/utility std ratio 0.0190, mean|bonus|/mean|u| 0.0224

## 6. Novelty Analysis
- PHEME: dynamic-only novelty 0.5654 vs removed static 0.0932 (gain +0.4722); utility delta -0.2267
- Ma-Weibo: dynamic-only novelty 0.2545 vs removed static 0.0911 (gain +0.1634); utility delta -0.1048

## 7. Persistence Analysis
- PHEME: P(select|persistent) 0.8718 vs P(select|new) 0.3123 (lift +0.5596), saved-evidence transitions {'wrong_to_correct': 10, 'correct_to_wrong': 2, 'both_wrong': 226, 'both_correct': 1228, 'net_correction_gain': 8}
- Ma-Weibo: P(select|persistent) 0.6753 vs P(select|new) 0.0342 (lift +0.6412), saved-evidence transitions {'wrong_to_correct': 4, 'correct_to_wrong': 9, 'both_wrong': 267, 'both_correct': 5436, 'net_correction_gain': -5}

## 8. Stage-wise Analysis
- PHEME very_early: delta +0.0000, ranking-change 0.1485, selection-change 0.0291, pred-change 0.0001
- PHEME early: delta +0.0000, ranking-change 0.3213, selection-change 0.0959, pred-change 0.0008
- PHEME mid: delta +0.0003, ranking-change 0.3860, selection-change 0.1474, pred-change 0.0010
- Ma-Weibo very_early: delta -0.0000, ranking-change 0.3300, selection-change 0.0610, pred-change 0.0002
- Ma-Weibo early: delta +0.0000, ranking-change 0.6863, selection-change 0.1256, pred-change 0.0007
- Ma-Weibo mid: delta +0.0001, ranking-change 0.7610, selection-change 0.1533, pred-change 0.0005

## 9. Selection-pressure Analysis
- PHEME low-pressure (n=58318): delta -0.0000, selection-change 0.0048, pred-change 0.0000
- PHEME medium-pressure (n=19922): delta -0.0001, selection-change 0.1756, pred-change 0.0013
- PHEME high-pressure (n=15864): delta +0.0009, selection-change 0.3243, pred-change 0.0022
- Ma-Weibo low-pressure (n=15769): delta +0.0001, selection-change 0.0044, pred-change 0.0001
- Ma-Weibo medium-pressure (n=9195): delta -0.0001, selection-change 0.0932, pred-change 0.0003
- Ma-Weibo high-pressure (n=54209): delta +0.0000, selection-change 0.1506, pred-change 0.0006

## 10. Validation-to-Test Stability
- PHEME fold0: val top1-top5 gap 0.00026, test delta +0.00022 FLAT_VALIDATION_LANDSCAPE
- PHEME fold1: val top1-top5 gap 0.00014, test delta -0.00037 FLAT_VALIDATION_LANDSCAPE
- PHEME fold2: val top1-top5 gap 0.00028, test delta +0.00024 FLAT_VALIDATION_LANDSCAPE
- PHEME fold3: val top1-top5 gap 0.00000, test delta +0.00000 FLAT_VALIDATION_LANDSCAPE
- PHEME fold4: val top1-top5 gap 0.00025, test delta +0.00044 FLAT_VALIDATION_LANDSCAPE
- Ma-Weibo fold0: val top1-top5 gap 0.00000, test delta -0.00019 FLAT_VALIDATION_LANDSCAPE
- Ma-Weibo fold1: val top1-top5 gap 0.00046, test delta -0.00019 FLAT_VALIDATION_LANDSCAPE
- Ma-Weibo fold2: val top1-top5 gap 0.00015, test delta +0.00013 FLAT_VALIDATION_LANDSCAPE
- Ma-Weibo fold3: val top1-top5 gap 0.00015, test delta +0.00019 FLAT_VALIDATION_LANDSCAPE
- Ma-Weibo fold4: val top1-top5 gap 0.00015, test delta +0.00013 FLAT_VALIDATION_LANDSCAPE

## 11. Parameter Necessity
- PHEME: lambda_n=0 folds [2], lambda_p=0 folds [4], both>0 folds [0, 1, 3]
- Ma-Weibo: lambda_n=0 folds [4], lambda_p=0 folds [2], both>0 folds [0, 1, 3]

## Root Cause Ranking
1. DYNAMIC_RANKING_RARELY_CHANGES (pheme, signal=0.0392)
2. DYNAMIC_RANKING_RARELY_CHANGES (maweibo, signal=0.0190)
3. SELECTION_CHANGES_BUT_PROXY_INSENSITIVE (maweibo, signal=0.0043)

## Implications for Method Design
Diagnosis only; no method changes were made in this round.

## Recommendation
REDESIGN_DYNAMIC
