# PHEME Ablation Study for the Final UMER Model

## Fixed protocol

Every newly trained ablation uses the final Round32 PHEME protocol unchanged:

- five fixed outer folds with `partition_seed=3090`;
- `training_seed=2000`;
- clean DeBERTa-v3-large initialization;
- physical/effective event batch 4/64;
- square-root inverse-frequency class weights;
- two standard stochastic text views and generalized-JSD weight 0.5 unless
  that consistency term is the ablated component;
- linear three-evidence fusion, SAM `rho=0.05`, AdamW, the original learning
  rates and weight decay;
- `legacy_four_metric_min` validation-only checkpoint/threshold selection;
- `max_epochs=30`, `patience=7`.

The completed full Round32 model is not retrained.

## Ablation matrix

| ID | Report label | Isolated change |
|---|---|---|
| `no_graph` | w/o Graph Evidence | Skip the propagation graph branch, set its two fusion logits to zero, and set graph auxiliary loss weight to zero. |
| `no_text` | w/o Text Evidence | Skip DeBERTa, set its two fusion logits and all view logits to zero, set text auxiliary and consistency weights to zero. |
| `no_memory` | w/o Cross-Event Memory | Skip retrieval and set the scalar memory logit to zero. |
| `no_consistency` | w/o Cross-View Consistency | Keep both stochastic views and their classification/fusion averaging, but set only generalized-JSD weight to zero. |
| `no_sam` | w/o SAM | Reuse the completed Round21 five-fold result because Round21 is configuration-identical except `sam_rho=0`. |

Zeroing an absent evidence slot preserves the same linear fusion-head shape
across ablations, so differences cannot be attributed to a different fusion
parameterization. An ablated neural branch is not executed and receives no
auxiliary supervision.

## Outputs

New results are isolated under:

`results/round_032_sam/ablation_pheme_seed3090_2000/<ablation>/`

Each ablation contains five fold directories, per-fold logs, and `summary.json`.
The queue never writes into the completed full-model or Round21 directories.
