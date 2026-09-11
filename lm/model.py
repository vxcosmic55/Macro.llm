# lm/model.py
"""
Decoder-only transformer, pure PyTorch. Pre-norm, RMSNorm, rotary embeddings,
grouped-query attention via F.scaled_dot_product_attention, SwiGLU MLP,
optional tied embeddings, activation checkpointing, and a KV cache for
generation.

Sanity check (no data needed):
    python -m lm.model --preset 100m --device cuda
"""
from __future__ import annotations

import argparse
import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lm.config import ModelConfig, add_model_args, config_from_args, param_count


# --------------------------------------------------------------------------- #
# Norm
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Normalization statistics in fp32 regardless of autocast dtype."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight


# --------------------------------------------------------------------------- #
# Rotary embeddings
# --------------------------------------------------------------------------- #
def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)              # (seq_len, head_dim//2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim//2). Interleaved pairs."""
    B, H, T, hd = x.shape
    dtype = x.dtype
    xf = x.float().view(B, H, T, hd // 2, 2)
    x1, x2 = xf[..., 0], xf[..., 1]                 # (B, H, T, hd//2)
    cos = cos.view(1, 1, T, hd // 2)
    sin = sin.view(1, 1, T, hd // 2)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(3).to(dtype)


# --------------------------------------------------------------------------- #
# KV cache (generation only)
# --------------------------------------------------------------------------- #
class KVCache:
    def __init__(self, n_layer: int, batch: int, n_kv_head: int, max_len: int,
                 head_dim: int, device: torch.device, dtype: torch.dtype):
        shape = (batch, n_kv_head, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.max_len = max_len
        self.pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        T = k.size(2)
        end = self.pos + T
        assert end <= self.max_len, f"KV cache overflow: {end} > {self.max_len}"
        self.k[layer_idx][:, :, self.pos:end] = k.to(self.k[layer_idx].dtype)
        self.v[layer_idx][:, :, self.pos:end] = v.to(self.v[layer_idx].dtype)
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def reset(self) -> None:
        self.pos = 0


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None, layer_idx: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)                 # (B, n_head,    T, hd)
        k = apply_rope(k, cos, sin)                 # (B, n_kv_head, T, hd)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)    # (B, n_kv_head, S, hd), S >= T
        k = k.to(q.dtype)
        v = v.to(q.dtype)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.size(2)
        p = self.dropout if self.training else 0.0
        if S == T:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        else:
            # Cached decode: queries sit at positions [S-T, S). is_causal would
            # align the mask top-left, which is wrong here, so build it explicitly.
            past = S - T
            qi = torch.arange(past, S, device=q.device).unsqueeze(1)   # (T, 1)
            ki = torch.arange(S, device=q.device).unsqueeze(0)         # (1, S)
            mask = ki <= qi                                            # (T, S) bool
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.wo(y)


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        x = x + self.resid_drop(self.attn(self.norm_attn(x), cos, sin, cache, layer_idx))
        x = x + self.resid_drop(self.mlp(self.norm_mlp(x)))
        return x


# --------------------------------------------------------------------------- #
# Chunked lm_head + cross entropy
# --------------------------------------------------------------------------- #
def _chunk_ce(x_chunk: torch.Tensor, weight: torch.Tensor,
              t_chunk: torch.Tensor, reduction: str) -> torch.Tensor:
    logits = F.linear(x_chunk, weight)
    return F.cross_entropy(logits.float(), t_chunk, ignore_index=-100,
                           reduction=reduction)


def lm_head_loss(x: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                 chunk_tokens: int, reduction: str = "mean") -> torch.Tensor:
    """
    x: (N, d_model) flattened hidden states, targets: (N,).
    Recomputes each chunk's logits in backward, so peak logit memory is
    chunk_tokens * vocab rather than N * vocab. At vocab 32768 and N = 32768
    that is the difference between ~4 GB and ~0.4 GB.
    """
    N = x.size(0)
    if chunk_tokens <= 0 or chunk_tokens >= N:
        return _chunk_ce(x, weight, targets, reduction)

    parts = []
    use_ckpt = torch.is_grad_enabled() and x.requires_grad
    sub = "sum" if reduction in ("mean", "sum") else "none"
    for i in range(0, N, chunk_tokens):
        xc, tc = x[i:i + chunk_tokens], targets[i:i + chunk_tokens]
        if use_ckpt:
            parts.append(checkpoint(_chunk_ce, xc, weight, tc, sub,
                                    use_reentrant=False))
        else:
            parts.append(_chunk_ce(xc, weight, tc, sub))
    if reduction == "none":
        return torch.cat(parts, dim=0)
    total = torch.stack(parts).sum()
    if reduction == "sum":
        return total
    n_valid = (targets != -100).sum().clamp(min=1)
    return total / n_valid


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.gradient_checkpointing = False

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim,
                                    cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down the residual-writing projections so residual variance does
        # not grow with depth (GPT-2 trick, extended to the MLP down-projection).
        std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        self.gradient_checkpointing = enable

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None, all_logits: bool = False,
                loss_reduction: str = "mean"):
        """
        idx:     (B, T) int64 token ids
        targets: (B, T) int64, -100 to ignore
        returns (logits, loss); logits is (B, T, V) if all_logits else (B, 1, V)
                when targets is None, else None unless all_logits.
        """
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.cfg.max_seq_len, (
            f"sequence position {start + T} exceeds max_seq_len {self.cfg.max_seq_len}")

        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]

        x = self.emb_drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, cache, i, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache, i)
        x = self.norm_f(x)

        if cache is not None:
            cache.pos += T

        loss = None
        logits = None
        if targets is not None:
            loss = lm_head_loss(
                x.reshape(-1, self.cfg.d_model),
                self.lm_head.weight,
                targets.reshape(-1),
                self.cfg.loss_chunk_tokens,
                reduction=loss_reduction,
            )
            if all_logits:
                logits = self.lm_head(x)
        else:
            logits = self.lm_head(x if all_logits else x[:, -1:, :])
        return logits, loss

    # ------------------------------------------------------------------ #
    def configure_optimizers(self, weight_decay: float, lr: float,
                             betas: tuple[float, float], device_type: str):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # torch >= 2.0: fused AdamW is a single CUDA kernel, ~10% step-time win.
        supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if (supports_fused and device_type == "cuda") else {}
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, **extra)
        return opt, sum(p.numel() for p in decay), sum(p.numel() for p in no_decay)

    def make_cache(self, batch: int, max_len: int, device: torch.device,
                   dtype: torch.dtype) -> KVCache:
        return KVCache(self.cfg.n_layer, batch, self.cfg.n_kv_head,
                       min(max_len, self.cfg.max_seq_len), self.cfg.head_dim,
                       device, dtype)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test(cfg: ModelConfig, device: torch.device) -> None:
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    counted = param_count(cfg)["total"]
    actual = model.num_params()
    print(f"params: counted {counted:,} | actual {actual:,} | match={counted == actual}")

    B, T = 2, 64
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    model.enable_gradient_checkpointing(True)
    model.train()
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                         enabled=(device.type == "cuda"))
    with amp:
        _, loss = model(idx, tgt)
    loss.backward()
    print(f"train loss {loss.item():.4f} (ln(V)={math.log(cfg.vocab_size):.4f}), "
          f"grads ok={all(p.grad is not None for p in model.parameters())}")

    model.eval()
    model.enable_gradient_checkpointing(False)
    with torch.no_grad(), amp:
        full, _ = model(idx, all_logits=True)
        cache = model.make_cache(B, T + 8, device,
                                 torch.bfloat16 if device.type == "cuda" else torch.float32)
        pre, _ = model(idx[:, :-1], cache=cache)
        step, _ = model(idx[:, -1:], cache=cache)
    err = (step[:, -1].float() - full[:, -1].float()).abs().max().item()
    print(f"cached vs full last-token logits max abs err: {err:.5f} "
          f"(expect < 0.05 in bf16, < 1e-4 in fp32)")
    print(f"shapes: full {tuple(full.shape)} step {tuple(step.shape)} cache_pos {cache.pos}")


def main() -> None:
    p = argparse.ArgumentParser()
    add_model_args(p)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = config_from_args(args)
    _self_test(cfg, torch.device(args.device))


if __name__ == "__main__":
    main()