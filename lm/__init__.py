# lm/__init__.py
"""From-scratch language model: pure-PyTorch internals, config-driven scale.

Attributes are imported lazily so that `python -m lm.config` (params, memory and
token-budget math) runs without pulling torch into the process.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "ModelConfig": "lm.config",
    "PRESETS": "lm.config",
    "TRAIN_DEFAULTS": "lm.config",
    "make_config": "lm.config",
    "param_count": "lm.config",
    "memory_report": "lm.config",
    "token_budget": "lm.config",
    "Transformer": "lm.model",
    "KVCache": "lm.model",
    "RMSNorm": "lm.model",
    "PackedDataset": "lm.data",
    "generate": "lm.sample",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'lm' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__