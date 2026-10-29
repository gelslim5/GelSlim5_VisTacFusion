"""Tiny YAML -> nested namespace config loader.

Usage:
    from vistacfusion.utils.config import load_config, merge_configs
    model_cfg = load_config('configs/model.yaml')
    cfg = merge_configs('configs/model.yaml', 'configs/train.yaml', 'configs/data.yaml')
    print(cfg.fusion_trunk.num_layers)   # attribute access
    print(cfg['fusion_trunk']['num_layers'])  # dict access also works
"""
from __future__ import annotations

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __init__(self, data=None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = self._wrap(v)

    @staticmethod
    def _wrap(v):
        if isinstance(v, dict):
            return Config(v)
        if isinstance(v, (list, tuple)):
            return type(v)(Config._wrap(x) for x in v)
        return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                out[k] = type(v)(x.to_dict() if isinstance(x, Config) else x for x in v)
            else:
                out[k] = v
        return out


def load_config(path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f) or {})


def merge_configs(*paths) -> Config:
    """Load several YAMLs into a single namespace. Top-level keys are mostly disjoint across
    files (model/train/data); a key shared by two files (e.g. ``image_size``) is allowed only
    if the values agree -- conflicting values are an error."""
    merged = {}
    for p in paths:
        cfg = load_config(p)
        for k, v in cfg.items():
            if k in merged and merged[k] != v:
                raise KeyError(
                    f"Conflicting top-level config key '{k}' across files ({p}): "
                    f"{merged[k]!r} != {v!r}"
                )
            merged[k] = v
    return Config(merged)
