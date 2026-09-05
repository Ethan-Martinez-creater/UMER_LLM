"""YAML configuration loader with attribute-style access."""
import re
import yaml
from pathlib import Path


class Config:
    """Wrapper that provides attribute access to nested dicts from YAML."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def load_config(config_path: str) -> Config:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = Config(data)
    _resolve_relative_paths(cfg, path.parent)
    return cfg


def _resolve_relative_paths(cfg: Config, config_dir: Path) -> None:
    """Resolve only known path fields relative to the project root."""
    project_root = config_dir.parent  # configs/ -> project root

    path_fields = {
        "data": {"label_file", "labels_file", "raw_dir", "output_dir"},
        "preprocess": {"frame_size_metadata"},
        "pretrained": {"text_embedding_model", "sentiment_model"},
    }

    for section_name, keys in path_fields.items():
        section = cfg.__dict__.get(section_name)
        if section is None:
            continue
        for key in keys:
            value = section.__dict__.get(key)
            if isinstance(value, str):
                section.__dict__[key] = _resolve_path_value(value, project_root)


def _resolve_path_value(value: str, project_root: Path) -> str:
    """Resolve a path string unless it is already absolute."""
    if _is_absolute_path(value):
        return value
    return str((project_root / value).resolve())


def _is_absolute_path(value: str) -> bool:
    """Return True for POSIX absolute paths and Windows drive paths."""
    return value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value) is not None
