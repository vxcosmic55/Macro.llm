# lm/config.py
"""
Model configuration, parameter counting, and memory / token-budget math.

Run:
    python -m lm.config --preset 300m --micro_bs 16 --seq_len 2048 --grad_accum 4
    python -m lm.config --all
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace


# --------------------------------------------------------------------------- #
# Model config
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4          # GQA; must divide n_head
    d_model: int = 1024
    d_ff: int = 2816            # SwiGLU hidden; keep a multiple of 256
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    loss_chunk_tokens: int = 4096   # chunked lm_head+CE; 0 disables chunking

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_kv_head

    def validate(self) -> "ModelConfig":
        assert self.d_model % self.n_head == 0, "d_model must divide by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must divide by n_kv_head"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.head_dim <= 256, "SDPA flash kernels cap head_dim at 256"
        assert self.vocab_size < 2 ** 16, "packing writes uint16 token ids"
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known}).validate()


# name -> overrides. Sizes are total params *with tied embeddings* at vocab 32768.
PRESETS: dict[str, dict] = {
    "100m": dict(d_model=768,  n_layer=12, n_head=12, n_kv_head=4, d_ff=2048),
    "300m": dict(d_model=1024, n_layer=24, n_head=16, n_kv_head=4, d_ff=2816),
    "650m": dict(d_model=1536, n_layer=24, n_head=12, n_kv_head=4, d_ff=4096),
    "1b":   dict(d_model=2048, n_layer=22, n_head=16, n_kv_head=4, d_ff=5632),
}


def make_config(preset: str = "300m", **overrides) -> ModelConfig:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    kwargs = dict(PRESETS[preset])
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return ModelConfig(**kwargs).validate()


# --------------------------------------------------------------------------- #
# Parameter counting (exact, matches model.py construction)
# --------------------------------------------------------------------------- #
def param_count(cfg: ModelConfig) -> dict:
    hd = cfg.head_dim
    emb = cfg.vocab_size * cfg.d_model
    attn = (
        cfg.d_model * cfg.n_head * hd          # wq
        + cfg.d_model * cfg.n_kv_head * hd     # wk
        + cfg.d_model * cfg.n_kv_head * hd     # wv
        + cfg.n_head * hd * cfg.d_model        # wo
    )
    mlp = 3 * cfg.d_model * cfg.d_ff           # gate, up, down
    norms = 2 * cfg.d_model                    # two RMSNorm weights per block
    per_layer = attn + mlp + norms
    body = cfg.n_layer * per_layer + cfg.d_model  # + final norm
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    return {
        "embedding": emb,
        "per_layer": per_layer,
        "body": body,
        "lm_head": head,
        "non_embedding": body,
        "total": emb + body + head,
    }


def flops_per_token(cfg: ModelConfig, seq_len: int) -> float:
    """Fwd+bwd FLOPs per token: 6*N_non_embed plus the quadratic attention term."""
    n = param_count(cfg)["non_embedding"]
    return 6.0 * n + 12.0 * cfg.n_layer * cfg.d_model * seq_len


# --------------------------------------------------------------------------- #
# Memory math
# --------------------------------------------------------------------------- #
GB = 1024 ** 3


def memory_report(
    cfg: ModelConfig,
    micro_bs: int,
    seq_len: int,
    grad_checkpointing: bool = True,
    master_dtype_bytes: int = 4,   # fp32 params + fp32 grads, bf16 autocast compute
    optim_bytes_per_param: int = 8,  # AdamW exp_avg + exp_avg_sq in fp32
) -> dict:
    n = param_count(cfg)["total"]
    params = n * master_dtype_bytes
    grads = n * master_dtype_bytes
    optim = n * optim_bytes_per_param

    tok = micro_bs * seq_len
    # bf16 residual-stream tensor saved at each block boundary
    ckpt_acts = cfg.n_layer * tok * cfg.d_model * 2
    # peak of recomputing one block (qkv + swiglu intermediates), bf16
    recompute = tok * (4 * cfg.d_model + 3 * cfg.d_ff) * 2
    if not grad_checkpointing:
        # ~8 saved bf16 tensors of width d_model + 3 of width d_ff, per layer
        ckpt_acts = cfg.n_layer * tok * (8 * cfg.d_model + 3 * cfg.d_ff) * 2
        recompute = 0

    # lm_head + cross entropy: bf16 logits + fp32 cast + fp32 log_softmax + grad
    chunk = cfg.loss_chunk_tokens if cfg.loss_chunk_tokens > 0 else tok
    chunk = min(chunk, tok)
    loss_mem = chunk * cfg.vocab_size * (2 + 4 + 4 + 2)

    total = params + grads + optim + ckpt_acts + recompute + loss_mem
    total_with_slack = total * 1.15  # allocator fragmentation + cuda context

    return {
        "params_M": n / 1e6,
        "params_GB": params / GB,
        "grads_GB": grads / GB,
        "optimizer_GB": optim / GB,
        "activations_GB": (ckpt_acts + recompute) / GB,
        "loss_head_GB": loss_mem / GB,
        "subtotal_GB": total / GB,
        "estimated_peak_GB": total_with_slack / GB,
    }


def token_budget(cfg: ModelConfig, tokens: int, seq_len: int,
                 achieved_tflops: float | None = None) -> dict:
    n = param_count(cfg)["total"]
    total_flops = flops_per_token(cfg, seq_len) * tokens
    out = {
        "params_M": n / 1e6,
        "tokens_B": tokens / 1e9,
        "tokens_per_param": tokens / n,
        "total_PFLOPs": total_flops / 1e15,
    }
    if achieved_tflops:
        hours = total_flops / (achieved_tflops * 1e12) / 3600
        out["gpu_hours_at_%.0fTFLOPs" % achieved_tflops] = hours
        out["gpu_days"] = hours / 24
    return out


# --------------------------------------------------------------------------- #
# Config layering: defaults < json file < CLI
# --------------------------------------------------------------------------- #
def load_json_config(path: str | None) -> dict:
    """A run config is {"model": {...}, "train": {...}}; both sections optional."""
    if not path:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    unknown = set(cfg) - {"model", "train", "notes"}
    if unknown:
        raise KeyError(f"{path}: unknown top-level sections {sorted(unknown)}")
    return cfg


def resolve_config(defaults: dict, file_cfg: dict, cli: dict) -> SimpleNamespace:
    merged = dict(defaults)
    for layer in (file_cfg, cli):
        unknown = set(layer) - set(defaults)
        if unknown:
            raise KeyError(f"unknown config keys {sorted(unknown)}; "
                           f"valid keys: {sorted(defaults)}")
        merged.update(layer)
    return SimpleNamespace(**merged)


# --------------------------------------------------------------------------- #
# argparse helpers shared by train/eval/sample
# --------------------------------------------------------------------------- #
# Training defaults live here so that configs/*.json, argparse and the memory
# report all read from one place. Every argparse default is None; "not passed"
# must stay distinguishable from "passed a value equal to the default".
TRAIN_DEFAULTS: dict = dict(
    data_dir="data/mix1",
    out_dir="runs/300m",
    seed=1337,
    device=None,
    seq_len=2048,
    micro_bs=16,
    grad_accum=4,
    total_tokens="15e9",
    max_steps=0,
    lr=3e-4,
    min_lr_frac=0.1,
    warmup_tokens="300e6",
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    log_interval=10,
    eval_interval=500,
    eval_batches=40,
    ckpt_interval=1000,
    keep_last=3,
    grad_checkpointing=True,
    compile=False,
    achieved_tflops=None,
)


def add_model_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--preset", type=str, default=None, choices=sorted(PRESETS))
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--rope_theta", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--loss_chunk_tokens", type=int, default=None)
    p.add_argument("--no_tie_embeddings", action="store_true", default=None)
    return p


def config_from_args(args: argparse.Namespace,
                     extra: dict | None = None) -> ModelConfig:
    """CLI flags win over `extra` (typically the "model" section of a json config)."""
    cli = dict(
        vocab_size=getattr(args, "vocab_size", None),
        d_model=getattr(args, "d_model", None),
        n_layer=getattr(args, "n_layer", None),
        n_head=getattr(args, "n_head", None),
        n_kv_head=getattr(args, "n_kv_head", None),
        d_ff=getattr(args, "d_ff", None),
        max_seq_len=getattr(args, "max_seq_len", None),
        rope_theta=getattr(args, "rope_theta", None),
        dropout=getattr(args, "dropout", None),
        loss_chunk_tokens=getattr(args, "loss_chunk_tokens", None),
    )
    cli = {k: v for k, v in cli.items() if v is not None}
    if getattr(args, "no_tie_embeddings", None):
        cli["tie_embeddings"] = False
    over = dict(extra or {})
    file_preset = over.pop("preset", None)
    preset = getattr(args, "preset", None) or file_preset or "300m"
    over.update(cli)
    unknown = set(over) - {f.name for f in fields(ModelConfig)}
    if unknown:
        raise KeyError(f"unknown model config keys {sorted(unknown)}")
    return make_config(preset, **over)


def _fmt(d: dict) -> str:
    return "\n".join(
        f"  {k:<28} {v:,.3f}" if isinstance(v, float) else f"  {k:<28} {v:,}"
        for k, v in d.items()
    )


def main() -> None:
    p = argparse.ArgumentParser(description="param / memory / token-budget report")
    add_model_args(p)
    p.add_argument("--micro_bs", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--tokens", type=float, default=15e9)
    p.add_argument("--achieved_tflops", type=float, default=None,
                   help="your measured bf16 throughput, for a wall-clock estimate")
    p.add_argument("--no_grad_checkpointing", action="store_true")
    p.add_argument("--all", action="store_true", help="report every preset")
    args = p.parse_args()

    names = sorted(PRESETS) if args.all else [args.preset or "300m"]
    for name in names:
        args.preset = name
        cfg = config_from_args(args)
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
        pc = param_count(cfg)
        mem = memory_report(cfg, args.micro_bs, args.seq_len,
                            grad_checkpointing=not args.no_grad_checkpointing)
        tb = token_budget(cfg, int(args.tokens), args.seq_len, args.achieved_tflops)
        eff = args.micro_bs * args.grad_accum
        print(f"\n=== preset {name} ===")
        print(json.dumps({k: v for k, v in cfg.to_dict().items()}, indent=2))
        print(f"params: total {pc['total']/1e6:.1f}M  "
              f"non-embedding {pc['non_embedding']/1e6:.1f}M  "
              f"embedding {pc['embedding']/1e6:.1f}M")
        print(f"effective batch: {eff} sequences = {eff*args.seq_len:,} tokens/step")
        print("memory:")
        print(_fmt(mem))
        print("token budget:")
        print(_fmt(tb))


if __name__ == "__main__":
    main()