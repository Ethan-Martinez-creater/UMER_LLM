# CR-TSER V2 artifact namespace

Formal V2 runs write here, so no V1 artifact under `results/cr_tser/` can ever
be overwritten (amendment V2 §22).

```text
results/cr_tser_v2/
├── p0/
├── manifests/
│   ├── maweibo/          # primary
│   └── pheme/            # secondary
├── utility_labels/
├── predictors/
├── unseen_reader/
├── reports/
└── verifier/
```

`results/cr_tser/` stays historical and read-only: it holds the V1 Weibo22
candidate-dataset P0 evidence (`results/cr_tser/p0/weibo22_*`), which the V2
protocol retains as a feasibility rejection rather than a method failure.

The primary dataset is Ma-Weibo and the secondary is PHEME. Weibo22 never
enters a V2 P1–P4 execution loop, and B2/S6 remain PHEME-only diagnostics.
