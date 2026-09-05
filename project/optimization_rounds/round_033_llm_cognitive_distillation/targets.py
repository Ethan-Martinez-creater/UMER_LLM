from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


EXPECTED_SCHEMA_ID = "umer_cognitive_event_target_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_aligned_cognitive_targets(event_ids, target_path, schema_path):
    target_path = Path(target_path)
    schema_path = Path(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema_id") != EXPECTED_SCHEMA_ID:
        raise ValueError("unexpected cognitive target schema")
    dimension = int(schema["target_dimension"])
    names = list(schema["target_names"])
    if dimension < 1 or len(names) != dimension or len(set(names)) != dimension:
        raise ValueError("invalid cognitive target dimension/names")

    event_ids = [str(event_id) for event_id in event_ids]
    event_index = {event_id: index for index, event_id in enumerate(event_ids)}
    if len(event_index) != len(event_ids):
        raise ValueError("duplicate event IDs in detector dataset")
    targets = torch.zeros((len(event_ids), dimension), dtype=torch.float32)
    masks = torch.zeros_like(targets)
    seen = set()
    unknown = []
    for line_number, line in enumerate(
        target_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = str(row["event_id"])
        if event_id in seen:
            raise ValueError(f"duplicate cognitive event at line {line_number}")
        seen.add(event_id)
        if event_id not in event_index:
            unknown.append(event_id)
            continue
        target = [float(value) for value in row["target"]]
        mask = [float(value) for value in row["target_mask"]]
        if len(target) != dimension or len(mask) != dimension:
            raise ValueError(f"cognitive dimension mismatch at line {line_number}")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in target):
            raise ValueError(f"cognitive target outside [0,1] at line {line_number}")
        if any(value not in (0.0, 1.0) for value in mask):
            raise ValueError(f"non-binary cognitive mask at line {line_number}")
        index = event_index[event_id]
        targets[index] = torch.tensor(target, dtype=torch.float32)
        masks[index] = torch.tensor(mask, dtype=torch.float32)
    if unknown:
        raise ValueError(f"cognitive targets contain {len(unknown)} unknown event IDs")
    report = {
        "schema_id": EXPECTED_SCHEMA_ID,
        "target_dimension": dimension,
        "target_names": names,
        "dataset_event_count": len(event_ids),
        "target_event_count": len(seen),
        "events_with_any_active_target": int((masks.sum(dim=1) > 0).sum()),
        "target_file_sha256": file_sha256(target_path),
        "schema_file_sha256": file_sha256(schema_path),
    }
    return targets, masks, report


def active_event_coverage(mask, indices) -> float:
    selected = mask[torch.as_tensor(indices, dtype=torch.long)]
    return float((selected.sum(dim=1) > 0).float().mean()) if len(indices) else 0.0
