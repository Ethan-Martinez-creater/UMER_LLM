#!/usr/bin/env python
"""Formal E1 finalization — UMER vs TC-DSCR fold-ID parity audit.

For every dataset x outer fold, restores the historical UMER split exactly
as the old training runner did and compares it with the TC-DSCR Protocol A
split:

- gold  — the old runner's own functions loaded verbatim from the historical
  server tree (screen_fold.ordered_labels + strict_indices over the file
  order of splits/all_event_ids.txt and the labels.csv manifest);
- sim   — the unit-tested local equivalent (tcdscr_fold_parity
  .reconstruct_old_umer_split), checked against gold for identity;
- new   — TC-DSCR build_primary_fold_split over sorted(event_label_registry).

PASS (per dataset x fold) requires exact train/validation/test set equality
and zero old-train -> new-test, old-train -> new-validation,
old-validation -> new-test overlap. Any failure marks
UMER_INIT_FOLD_PARITY_FAIL; no split or E1 result is modified.

Writes results/tcdscr/formal_e1_finalization/fold_parity_audit.json and
fold_parity_table.md.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

OLD_TREES = {
    "pheme": "/data/jyz/next/original_model_optimization",
    "maweibo": "/data/jyz/next/llm/project",
}
OLD_DATA_DIRS = {
    "pheme": "/data/jyz/next/modified/common/pheme_240h_data",
    "maweibo": "/data/jyz/next/llm/data/maweibo_240h_data",
}
FOLDS = (0, 1, 2, 3, 4)
PARTITION_SEED = 3090


def load_old_event_ids_labels(data_dir):
    """Historical order + label manifest, exactly as ordered_labels reads it
    (splits/all_event_ids.txt file order, labels.csv map)."""
    ids_path = os.path.join(data_dir, "splits", "all_event_ids.txt")
    with open(ids_path, encoding="utf-8") as fh:
        event_ids = [line.strip() for line in fh if line.strip()]
    label_map = {}
    with open(os.path.join(data_dir, "labels.csv"),
              encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label_map[str(row["event_id"]).strip()] = int(row["label"])
    missing = [e for e in event_ids if e not in label_map]
    if missing:
        raise ValueError(f"{len(missing)} event ids missing label: "
                         f"{missing[:5]}")
    labels = [label_map[e] for e in event_ids]
    return event_ids, labels


def load_gold_functions(dataset):
    """Load the old runner's ordered_labels/strict_indices from its tree."""
    screen = os.path.join(OLD_TREES[dataset], "optimization_rounds",
                          "round_006_node_deberta", "screen_fold.py")
    tree_src = os.path.join(OLD_TREES[dataset], "src")
    for p in (OLD_TREES[dataset], tree_src, os.path.join(OLD_TREES[dataset],
                                                         "protocol")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        f"umor_screen_fold_{dataset}", screen)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.ordered_labels, mod.strict_indices


def restore_gold_split(dataset, fold_index, partition_seed=PARTITION_SEED):
    ordered_labels, strict_indices = load_gold_functions(dataset)
    data_dir = Path(OLD_DATA_DIRS[dataset])  # old function expects a Path
    event_ids, labels, _frame = ordered_labels(data_dir)
    train_idx, val_idx, test_idx = strict_indices(
        labels, fold_index, partition_seed=partition_seed)
    return {"train": [event_ids[i] for i in train_idx],
            "validation": [event_ids[i] for i in val_idx],
            "test": [event_ids[i] for i in test_idx]}, labels


def run_audit(out_root):
    # import tcdscr_common first: it registers the project package on
    # sys.path for the deployed scripts/project layout
    import tcdscr_common  # noqa: F401
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import event_label_registry
    from tcdscr_fold_parity import compare_fold_parity, parity_pass

    os.makedirs(out_root, exist_ok=True)
    report = {"partition_seed": PARTITION_SEED,
              "validation_fraction": 0.10, "folds": {}, "datasets": {},
              "summary": {}}
    rows = []  # markdown rows
    all_pass = True
    any_fail = False
    for dataset in ("pheme", "maweibo"):
        cfg = config_from_env(dataset)
        registry = event_label_registry(dataset, cfg)
        old_ids, old_labels = load_old_event_ids_labels(
            OLD_DATA_DIRS[dataset])
        old_label_map = dict(zip(old_ids, old_labels))
        mism = [(e, old_label_map[e], registry[e]) for e in old_ids
                if e in registry and old_label_map[e] != registry[e]]
        report["datasets"][dataset] = {
            "events_old": len(old_ids),
            "events_registry": len(registry),
            "label_mismatches": len(mism),
            "label_mismatch_first10": mism[:10],
            "ids_only_in_old_first10":
                sorted(set(old_ids) - set(registry))[:10],
            "ids_only_in_registry_first10":
                sorted(set(registry) - set(old_ids))[:10],
        }
        for fold in FOLDS:
            gold, gold_labels = restore_gold_split(dataset, fold)
            # sim: unit-tested local equivalent, must equal gold
            from tcdscr_fold_parity import reconstruct_old_umer_split
            sim = reconstruct_old_umer_split(old_ids, old_labels, fold)
            identical_gold_sim = (
                sim["train"] == gold["train"]
                and sim["validation"] == gold["validation"]
                and sim["test"] == gold["test"])
            new = build_primary_fold_split(registry, fold, seed=3090)
            cmp = compare_fold_parity(gold, new)
            cmp["gold_vs_sim_identical"] = identical_gold_sim
            ok = parity_pass(cmp) and identical_gold_sim
            cmp["parity_pass"] = ok
            all_pass &= ok
            if not ok:
                any_fail = True
            report["folds"].setdefault(dataset, {})[str(fold)] = cmp
            rows.append({
                "dataset": dataset, "fold": fold,
                "train_exact": cmp["train_exact_match"],
                "val_exact": cmp["validation_exact_match"],
                "test_exact": cmp["test_exact_match"],
                "old_tr_n_new_te": cmp["old_train_intersect_new_test"],
                "old_tr_n_new_va": cmp["old_train_intersect_new_validation"],
                "old_va_n_new_te": cmp["old_validation_intersect_new_test"],
                "gold_eq_sim": identical_gold_sim,
                "parity_pass": ok,
            })
    report["summary"] = {
        "all_folds_pass": all_pass,
        "status": "PASS" if all_pass else "UMER_INIT_FOLD_PARITY_FAIL",
        "n_folds": 2 * len(FOLDS),
        "n_pass": sum(1 for r in rows if r["parity_pass"]),
        "note": ("gold = old runner functions run verbatim from the "
                 "historical tree; sim = unit-tested local equivalent; "
                 "new = TC-DSCR build_primary_fold_split over the sorted "
                 "event registry"),
    }
    with open(os.path.join(out_root, "fold_parity_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    lines = ["# Fold-ID parity audit — old UMER vs TC-DSCR (E1 finalization)",
             "",
             f"partition_seed={PARTITION_SEED}, 5 outer folds, 10% inner "
             "validation. gold = historical runner functions on the old "
             "event ordering/labels; new = TC-DSCR Protocol A on the sorted "
             "TC-DSCR registry. PASS = exact match on all three sets and "
             "zero cross-set overlap.",
             "", "| dataset | fold | train exact | val exact | test exact | "
             "old-trn ∩ new-tst | old-trn ∩ new-val | old-val ∩ new-tst | "
             "gold==sim | PASS |",
             "|---|---|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['fold']} | {r['train_exact']} | "
            f"{r['val_exact']} | {r['test_exact']} | "
            f"{r['old_tr_n_new_te']} | {r['old_tr_n_new_va']} | "
            f"{r['old_va_n_new_te']} | {r['gold_eq_sim']} | "
            f"{r['parity_pass']} |")
    lines += ["", f"Overall: {report['summary']['status']}",
              f"- folds passing: {report['summary']['n_pass']} / "
              f"{report['summary']['n_folds']}", ""]
    with open(os.path.join(out_root, "fold_parity_table.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(json.dumps(report["summary"], indent=1))
    print("wrote", os.path.join(out_root, "fold_parity_audit.json"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/data/jyz/next/llm/results/"
                                          "tcdscr/formal_e1_finalization")
    args = ap.parse_args()
    sys.exit(run_audit(args.out_root))