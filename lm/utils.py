# lm/utils.py
"""
Runtime helpers: seeding, device and autocast selection, atomic checkpoint IO,
and jsonl logging.

Nothing in here knows about the model architecture. Config layering lives in
lm/config.py so that the params/memory report stays importable without torch.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(device: torch.device):
    """Returns (context manager, enabled). bf16 only; no GradScaler needed."""
    enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)
    return ctx, enabled


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
CKPT_PREFIX, CKPT_SUFFIX = "ckpt_", ".pt"


def numbered_ckpts(out_dir: str) -> list[str]:
    """Step checkpoints only, sorted; ckpt_best.pt must never be rotated away."""
    if not os.path.isdir(out_dir):
        return []
    return sorted(f for f in os.listdir(out_dir)
                  if f.startswith(CKPT_PREFIX) and f.endswith(CKPT_SUFFIX)
                  and f[len(CKPT_PREFIX):-len(CKPT_SUFFIX)].isdigit())


def latest_ckpt(out_dir: str) -> str:
    ck = numbered_ckpts(out_dir)
    if not ck:
        raise FileNotFoundError(f"no {CKPT_PREFIX}*{CKPT_SUFFIX} in {out_dir}")
    return os.path.join(out_dir, ck[-1])


def prune_ckpts(out_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    for f in numbered_ckpts(out_dir)[:-keep]:
        os.remove(os.path.join(out_dir, f))


def save_ckpt(path: str, raw_model, optimizer, model_cfg_dict: dict, step: int,
              tokens_seen: int, run_cfg: dict, best_val: float) -> None:
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model_cfg_dict,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "run_config": run_cfg,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }, tmp)
    os.replace(tmp, path)  # atomic: a killed job never leaves a half-written ckpt


def load_ckpt(path: str, map_location="cpu") -> dict:
    # torch >= 2.6 defaults weights_only=True; these checkpoints carry RNG state
    # and the run config, so the full unpickler is required.
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rng(ckpt: dict) -> None:
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])


def strip_compile_prefix(state: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a")

    def write(self, record: dict, echo: bool = True) -> None:
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        if echo:
            print(" ".join(f"{k}={v}" for k, v in record.items()), flush=True)

    def close(self) -> None:
        self.f.close()