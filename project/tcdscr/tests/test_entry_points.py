"""Unit tests: training entry points are fold-aware (delta-fix §7/§36)."""
import pytest

tcdscr_train_encoder = pytest.importorskip(
    "tcdscr_train_encoder", reason="entry-point scripts not on sys.path")
tcdscr_train_selector = pytest.importorskip(
    "tcdscr_train_selector", reason="entry-point scripts not on sys.path")


def test_train_encoder_fold_argument():
    parser = tcdscr_train_encoder.build_parser()
    args = parser.parse_args(["--dataset", "pheme", "--fold", "3",
                              "--partition-seed", "3090", "--epochs", "1"])
    assert args.fold == 3
    assert args.partition_seed == 3090
    assert args.dataset == "pheme"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "pheme"])  # fold is required


def test_train_selector_fold_argument():
    parser = tcdscr_train_selector.build_parser()
    args = parser.parse_args(["--dataset", "maweibo", "--fold", "0",
                              "--partition-seed", "3090"])
    assert args.fold == 0
    assert args.partition_seed == 3090
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "maweibo", "--fold", "9"])


def test_train_entries_reject_out_of_range_fold():
    from tcdscr.data.temporal_split import build_primary_fold_split
    labels = {f"e{i}": i % 2 for i in range(50)}
    with pytest.raises(ValueError):
        build_primary_fold_split(labels, fold_index=5)
    with pytest.raises(ValueError):
        build_primary_fold_split(labels, fold_index=-1)


def test_split_manifest_shape():
    """make_split_manifest emits every field required by delta-fix §7.3."""
    from tcdscr.data.temporal_split import build_primary_fold_split
    labels = {f"e{i}": i % 2 for i in range(100)}
    split = build_primary_fold_split(labels, fold_index=1, seed=3090)
    manifest = tcdscr_train_encoder.make_split_manifest(
        "pheme", 1, 3090, split, labels)
    for key in ("dataset", "fold", "partition_seed", "train_event_ids",
                "validation_event_ids", "test_event_ids",
                "train_label_counts", "validation_label_counts",
                "test_label_counts"):
        assert key in manifest
    assert manifest["fold"] == 1
    assert manifest["partition_seed"] == 3090
    assert (set(manifest["train_event_ids"])
            & set(manifest["test_event_ids"])) == set()
