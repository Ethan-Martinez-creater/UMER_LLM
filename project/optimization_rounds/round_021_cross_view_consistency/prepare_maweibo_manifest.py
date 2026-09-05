"""Prepare and validate the read-only Ma-Weibo data manifest for Round21."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_labels(path: Path) -> dict[str, int]:
    labels = {}
    pattern = re.compile(r"eid:(\d+)\s+label:(\d+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            labels[match.group(1)] = int(match.group(2))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    event_ids = [
        value.strip()
        for value in (args.data_dir / "splits" / "all_event_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if value.strip()
    ]
    labels = parse_labels(args.label_file)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate event IDs in all_event_ids.txt")

    missing_labels = [event_id for event_id in event_ids if event_id not in labels]
    missing_raw = [
        event_id for event_id in event_ids
        if not (args.raw_dir / f"{event_id}.json").is_file()
    ]
    graph_dir = args.data_dir / "graph_final"
    missing_graphs = [
        event_id for event_id in event_ids
        if not (graph_dir / f"{event_id}.pt").is_file()
    ]
    if missing_labels or missing_raw or missing_graphs:
        raise FileNotFoundError(
            "manifest mismatch: "
            f"labels={len(missing_labels)}, raw={len(missing_raw)}, "
            f"graphs={len(missing_graphs)}"
        )

    output = args.data_dir / "labels.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "label"])
        writer.writerows((event_id, labels[event_id]) for event_id in event_ids)

    counts = {label: 0 for label in sorted(set(labels.values()))}
    for event_id in event_ids:
        counts[labels[event_id]] = counts.get(labels[event_id], 0) + 1
    print(
        f"validated {len(event_ids)} events; label_counts={counts}; "
        f"manifest={output}"
    )


if __name__ == "__main__":
    main()
