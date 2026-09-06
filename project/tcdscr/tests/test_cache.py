"""Unit tests: LLM cache keys (plan §28 — test_cache_key_deterministic)."""
import os

from ..llm.cache import LLMResponseCache, build_cache_key, sha256_text


def _key(prompt, **overrides):
    args = dict(
        model_id="/models/qwen3-8b", model_revision="main",
        prompt=prompt, generation_config={"do_sample": False},
        dataset="pheme", event_id="e1", cutoff=60,
        selector_checkpoint_hash="abc123", budget=512)
    args.update(overrides)
    return build_cache_key(**args)


def test_cache_key_deterministic():
    a = _key("same prompt")
    b = _key("same prompt")
    assert a == b
    # every frozen field participates in the key
    assert _key("other prompt") != a
    assert _key("same prompt", event_id="e2") != a
    assert _key("same prompt", cutoff=360) != a
    assert _key("same prompt", budget=1024) != a
    assert _key("same prompt", dataset="maweibo") != a
    assert _key("same prompt", selector_checkpoint_hash="xyz") != a
    assert _key("same prompt", generation_config={"do_sample": True}) != a
    assert _key("same prompt", model_revision="v2") != a


def test_same_key_never_inferred_twice(tmp_path):
    path = os.path.join(str(tmp_path), "llm_cache.jsonl")
    cache = LLMResponseCache(path)
    key = _key("prompt")
    cache.put(key, "RUMOR")
    cache.put(key, "NON_RUMOR")  # same key must not overwrite or re-run
    reloaded = LLMResponseCache(path)
    assert reloaded.get(key) == "RUMOR"
    assert reloaded.get(sha256_text("missing")) is None


def test_cache_key_field_set_frozen():
    from ..llm.cache import KEY_FIELDS
    assert KEY_FIELDS == (
        "model_id", "model_revision", "prompt_sha256",
        "generation_config_sha256", "dataset", "event_id", "cutoff",
        "selector_checkpoint_hash", "budget")
