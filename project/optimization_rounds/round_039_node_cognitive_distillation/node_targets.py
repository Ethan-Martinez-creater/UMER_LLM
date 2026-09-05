from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


EXPECTED_SCHEMA_ID = "umer_node_cognitive_target_v1"
STANCE_NAMES = ["source", "support", "deny", "query", "comment", "unclear"]
ROLE_NAMES = [
    "claim", "evidence", "correction", "reasoning", "emotion", "comment", "other"
]
TARGET_DIMENSION = len(STANCE_NAMES) + len(ROLE_NAMES) + 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_aligned_node_targets(event_ids, target_path, schema_path):
    target_path = Path(target_path)
    schema_path = Path(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema_id") != EXPECTED_SCHEMA_ID:
        raise ValueError("unexpected node cognitive target schema")
    if schema.get("stance_names") != STANCE_NAMES:
        raise ValueError("unexpected node cognitive stance ontology")
    if schema.get("role_names") != ROLE_NAMES:
        raise ValueError("unexpected node cognitive role ontology")
    if int(schema.get("target_dimension", -1)) != TARGET_DIMENSION:
        raise ValueError("unexpected node cognitive target dimension")

    event_ids = [str(event_id) for event_id in event_ids]
    event_index = {event_id: index for index, event_id in enumerate(event_ids)}
    if len(event_index) != len(event_ids):
        raise ValueError("duplicate event IDs in detector dataset")
    aligned = [None] * len(event_ids)
    seen = set()
    unknown = []
    active_node_count = 0
    for line_number, line in enumerate(
        target_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = str(row["event_id"])
        if event_id in seen:
            raise ValueError(f"duplicate node cognitive event at line {line_number}")
        seen.add(event_id)
        if event_id not in event_index:
            unknown.append(event_id)
            continue
        nodes = []
        positions = set()
        for node in row.get("nodes", []):
            position = int(node["original_position"])
            if position < 0 or position in positions:
                raise ValueError(f"invalid/duplicate node position at line {line_number}")
            positions.add(position)
            stance = int(node["stance"])
            role = int(node["role"])
            salience = float(node["salience"])
            stance_mask = bool(node["stance_mask"])
            role_mask = bool(node["role_mask"])
            salience_mask = bool(node["salience_mask"])
            if not 0 <= stance < len(STANCE_NAMES):
                raise ValueError(f"stance outside ontology at line {line_number}")
            if not 0 <= role < len(ROLE_NAMES):
                raise ValueError(f"role outside ontology at line {line_number}")
            if not 0.0 <= salience <= 1.0:
                raise ValueError(f"salience outside [0,1] at line {line_number}")
            if stance_mask or role_mask or salience_mask:
                active_node_count += 1
            nodes.append(
                {
                    "original_position": position,
                    "stance": stance,
                    "stance_mask": stance_mask,
                    "role": role,
                    "role_mask": role_mask,
                    "salience": salience,
                    "salience_mask": salience_mask,
                }
            )
        if nodes:
            aligned[event_index[event_id]] = nodes
    if unknown:
        raise ValueError(f"node targets contain {len(unknown)} unknown event IDs")
    report = {
        "schema_id": EXPECTED_SCHEMA_ID,
        "target_dimension": TARGET_DIMENSION,
        "dataset_event_count": len(event_ids),
        "target_event_count": len(seen),
        "events_with_nodes": sum(nodes is not None for nodes in aligned),
        "active_node_count": active_node_count,
        "target_file_sha256": file_sha256(target_path),
        "schema_file_sha256": file_sha256(schema_path),
    }
    return aligned, report


def active_event_coverage(aligned, indices) -> float:
    indices = [int(index) for index in indices]
    if not indices:
        return 0.0
    return sum(aligned[index] is not None for index in indices) / len(indices)


def pack_node_targets(aligned, indices, max_nodes, device):
    indices = [int(index) for index in indices]
    shape = (len(indices), int(max_nodes))
    packed = {
        "stance": torch.zeros(shape, dtype=torch.long, device=device),
        "stance_mask": torch.zeros(shape, dtype=torch.bool, device=device),
        "role": torch.zeros(shape, dtype=torch.long, device=device),
        "role_mask": torch.zeros(shape, dtype=torch.bool, device=device),
        "salience": torch.zeros(shape, dtype=torch.float32, device=device),
        "salience_mask": torch.zeros(shape, dtype=torch.bool, device=device),
    }
    for batch_index, dataset_index in enumerate(indices):
        nodes = aligned[dataset_index]
        if nodes is None:
            continue
        for node in nodes:
            position = int(node["original_position"])
            if position >= max_nodes:
                raise ValueError(
                    f"teacher node position {position} exceeds batch graph nodes {max_nodes}"
                )
            for name in packed:
                packed[name][batch_index, position] = node[name]
    return packed

