"""M1-F report rendering (``M1F_REPORT.md``).

Pure string assembly over the analysis payloads, so the report can be
regenerated from the JSON artifacts without recomputing anything.
"""
from __future__ import annotations

from ..config import protocol as P


def _f(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value != value:  # NaN
        return "nan"
    return f"{value:+.{digits}f}"


def _plain(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def _ci(record) -> str:
    if not record:
        return "n/a"
    return (f"[{_f(record.get('ci_low'))}, {_f(record.get('ci_high'))}]")


def build_report(data: dict) -> str:
    lines = [
        "# BCR-Utility M1-F — Strong-Baseline & Validity Audit Report",
        "",
        "```text",
        f"protocol           = {P.PROTOCOL_VERSION}",
        f"baseline commit    = {P.M1F_BASELINE_COMMIT}",
        f"verified commit    = {data.get('commit')}",
        f"environment        = {data.get('environment') or 'LOCAL'}",
        f"STATUS             = {data['verdict']['verdict']}",
        "```",
        "",
        "M1-F is a **bounded falsification round**. It reuses frozen M1/M1-E "
        "predictions, features and label caches only: no reader was loaded, no "
        "utility label was generated, no E0/E1/E2/E3 cache was regenerated and "
        "M2 was not entered. The frozen M1 and M1-E verdicts are unchanged.",
        "",
        "## 1. Task validity (Task A)",
        "",
    ]
    tv = data["task_validity"]
    lines += [
        "```text",
        f"relabelled historical data = {tv['relabelled_historical_data']}",
        f"utility is truth verification = {tv['utility_meaning']['is_verification_utility']}",
        f"claim allowed  = {tv['utility_meaning']['claim_allowed']}",
        f"claim forbidden = {tv['utility_meaning']['claim_forbidden']}",
        f"validity blocked = {tv['conclusion']['validity_blocked']}",
        "```",
        "",
        f"{tv['conclusion']['reason']}.",
        "",
        "Per dataset:",
        "",
    ]
    for dataset, spec in tv["datasets"].items():
        lines.append(f"* **{dataset}** — {spec['label_semantics']}; "
                     f"{spec['boundary']}.")
    lines += [
        "",
        f"Cross-dataset boundary: language-only shift = "
        f"**{tv['cross_dataset_boundary']['is_language_only_shift']}**. "
        f"{tv['cross_dataset_boundary']['note']}.",
        "",
        "## 2. Strong baselines (Task B)",
        "",
    ]
    for dataset in sorted(data["baselines"]):
        payload = data["baselines"][dataset]
        lines += [f"### {dataset}", "",
                  "| variant | kind | " +
                  " | ".join(P.READER_KEYS) + " | mean |",
                  "|---|---|" + "---|" * (len(P.READER_KEYS) + 1)]
        for variant in P.M1F_VARIANTS:
            entry = payload["variants"][variant]
            cells = [_plain(entry["per_reader_macro_f1"].get(reader))
                     for reader in P.READER_KEYS]
            mean = _plain(sum(entry["per_reader_macro_f1"].values())
                          / max(len(entry["per_reader_macro_f1"]), 1))
            lines.append(f"| {variant} | {entry['kind']} | "
                         + " | ".join(cells) + f" | {mean} |")
        comparator = payload["comparator"]
        lines += [
            "",
            "```text",
            f"dev-selected strong comparator = {comparator['selected']}",
            "selection = " + comparator["selected_by"],
            "```",
            "",
            "Dev Macro-F1 means (`utility_dev`, training readers only):",
            "",
        ]
        for variant in P.M1F_COMPARATOR_POOL:
            lines.append(f"* {variant}: {_plain(comparator['dev_macro_f1_mean'][variant])}")
        lines.append("")
    lines += ["## 3. Primary comparison (Task C)", ""]
    active = data["active"]
    primary = active["primary"]
    aggregate = primary["aggregate"]
    lines += [
        "```text",
        f"primary    = {primary['primary']}",
        f"comparator = {primary['comparator']}",
        f"mean delta Macro-F1 = {_f(aggregate['mean_delta_macro_f1'])}  "
        f"95% CI {_ci(aggregate)}",
        f"positive readers    = {aggregate['positive_readers']}/3",
        f"worst reader delta  = {_f(aggregate['worst_reader_delta'])}",
        "```",
        "",
        "Per held-out reader:",
        "",
        "| held-out | Δ Macro-F1 | 95% CI |",
        "|---|---|---|",
    ]
    for held in P.READER_KEYS:
        record = primary["per_reader_macro_f1_delta"].get(held)
        if record:
            lines.append(f"| {held} | {_f(record['observed'])} | {_ci(record)} |")
    lines += ["", "Secondary metric differences (event-clustered bootstrap):", "",
              "| metric | Δ | 95% CI |", "|---|---|---|"]
    for name, record in primary["secondary_metric_deltas"].items():
        lines.append(f"| {name} | {_f(record['observed'])} | {_ci(record)} |")
    lines += ["", "## 4. Active HELPFUL/HARMFUL audit (Task D)", "",
              "| variant | active n | H-vs-H Macro-F1 | balanced acc | "
              "HARMFUL AUPRC | HELPFUL AUPRC | NEUTRAL rate on active |",
              "|---|---|---|---|---|---|---|"]
    for variant in P.M1F_VARIANTS:
        act = active["variants"][variant]["active"]
        lines.append(
            f"| {variant} | {act.get('n_active')} | "
            f"{_plain(act.get('helpful_vs_harmful_macro_f1'))} | "
            f"{_plain(act.get('balanced_accuracy'))} | "
            f"{_plain(act.get('harmful_auprc'))} | "
            f"{_plain(act.get('helpful_auprc'))} | "
            f"{_plain(act.get('neutral_prediction_rate'))} |")
    lines += [
        "",
        "A NEUTRAL prediction on a gold-active row counts as an error and is "
        "kept in every denominator above.",
        "",
        "## 5. Within-snapshot evidence discrimination (Task E)",
        "",
        "| variant | snapshots used | rows used | mean snapshot ρ | "
        "centered pooled ρ | 95% CI | pairwise concordance |",
        "|---|---|---|---|---|---|---|",
    ]
    within = data["within"]
    for variant in P.M1F_TRAINED_VARIANTS + (P.M1F_S4,):
        entry = within["variants"][variant]
        ci = entry["centered_spearman_ci"]
        lines.append(
            f"| {variant} | {entry['n_snapshots_used']}/{entry['n_snapshots_total']} | "
            f"{entry['n_rows_used']} | "
            f"{_plain(entry['mean_within_snapshot_spearman'])} | "
            f"{_plain(entry['centered_pooled_spearman'])} | "
            f"[{_f(ci['ci_low'])}, {_f(ci['ci_high'])}] | "
            f"{_plain(entry.get('pairwise_concordance'))} |")
    lines += ["", "Centered-Spearman increment over S2:", "",
              "| variant | Δ centered ρ | 95% CI |", "|---|---|---|"]
    for variant, record in within["comparisons_vs_S2"].items():
        lines.append(f"| {variant} | {_f(record['observed'])} | {_ci(record)} |")
    lines += ["", "## 6. Fixed threshold diagnostics (Task F)", "",
              "```text",
              f"official threshold = +-{data['robustness']['official_threshold']}",
              "diagnostic only    = "
              + ", ".join(f"+-{t:.2f}" for t in data['robustness']['thresholds']),
              "threshold selected = "
              f"{data['robustness']['threshold_selected']}",
              "```",
              "",
              "Diagnostic Macro-F1 at each fixed threshold (primary S4/D5):",
              "",
              "| threshold | Macro-F1 | balanced acc | predicted activity |",
              "|---|---|---|---|"]
    for key in sorted(data["robustness"]["diagnostics"]):
        record = data["robustness"]["diagnostics"][key].get(P.M1F_S4) or {}
        lines.append(f"| ±{key} | {_plain(record.get('macro_f1'))} | "
                     f"{_plain(record.get('balanced_accuracy'))} | "
                     f"{_plain(record.get('predicted_activity_rate'))} |")
    activity = data["robustness"]["activity"]
    lines += [
        "",
        "```text",
        f"gold activity rate            = "
        f"{_plain(activity['gold_activity_rate'].get(P.M1F_S4))}",
        f"predicted activity rate       = "
        f"{_plain(activity['predicted_activity_rate'].get(P.M1F_S4))}",
        f"near-boundary rate of |u_hat| = "
        f"{_plain(activity['near_boundary_rate'].get(P.M1F_S4))}",
        f"near-boundary band            = +-{activity['near_boundary_threshold']} "
        f"+- {activity['band']}",
        f"A/B label swap                = "
        f"{data['robustness']['ab_label_swap']['status']}",
        "```",
        "",
        "## 7. PHEME diagnostic (Task G)",
        "",
        "```text",
    ]
    pheme_primary = data["pheme"]["primary_comparison"]
    lines += [
        f"PHEME is diagnostic_only = {data['pheme']['diagnostic_only']}",
        f"PHEME decides the gate   = {data['pheme']['decides_gate']}",
        f"comparator               = {pheme_primary['comparator']}",
        f"mean delta Macro-F1      = "
        f"{_f(pheme_primary['aggregate']['mean_delta_macro_f1'])} 95% CI "
        f"{_ci(pheme_primary['aggregate'])}",
        "```",
        "",
        data["pheme"]["note"],
        "",
        "## 8. M1-F decision",
        "",
        "```text",
    ]
    verdict = data["verdict"]
    for name, ok in verdict["checks"].items():
        lines.append(f"{name:58s} = {'PASS' if ok else 'FAIL'}")
    lines += [
        f"VERDICT = {verdict['verdict']}",
        "```",
        "",
        verdict["note"],
        "",
        "## 9. Immutability",
        "",
        "```text",
        f"frozen input groups pinned = {data['pins']['n_groups']} "
        f"({data['pins']['n_artifacts']} artifacts)",
        f"M1 verdict unchanged        = {verdict['m1_verdict_unchanged']}",
        f"M1-E verdict unchanged      = {verdict['m1e_verdict_unchanged']}",
        f"M2 entered                  = {verdict['m2_entered']}",
        "results/cr_tser{, _v2, _v2r1} = no tracked changes",
        "new reader inference        = none",
        "new utility labels          = none",
        "regenerated E0/E1/E2/E3     = none",
        "new model deployment        = none (Qwen3-0.6B / Phi / Gemma absent)",
        "```",
        "",
    ]
    return "\n".join(lines)
