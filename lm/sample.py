# lm/sample.py
"""
Load a checkpoint and generate text with a KV cache.

Run:
    python -m lm.sample --ckpt runs/300m/ckpt_best.pt --data_dir data/mix1 \
        --prompt "Q: If 3x + 7 = 22, what is x?\nA:" \
        --max_new_tokens 200 --temperature 0.8 --top_p 0.95 --num_samples 3
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from lm.config import ModelConfig
from lm.model import Transformer
from lm.utils import (autocast_ctx, load_ckpt, pick_device, set_seed,
                      strip_compile_prefix)


# --------------------------------------------------------------------------- #
def load_model(ckpt_path: str, device: torch.device):
    ckpt = load_ckpt(ckpt_path)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = Transformer(cfg)
    state = strip_compile_prefix(ckpt["model"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if cfg.tie_embeddings:
        missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    model.to(device).eval()
    return model, cfg, ckpt


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """logits: (B, V) float32 -> masked in place-safe fashion."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = probs - F.softmax(sorted_logits, dim=-1) >= top_p  # keep the token that crosses p
        drop[:, 0] = False
        mask = torch.zeros_like(drop).scatter_(1, sorted_idx, drop)
        logits = logits.masked_fill(mask, float("-inf"))
    return logits


@torch.no_grad()
def generate(model: Transformer, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0,
             repetition_penalty: float = 1.0, eos_id: int | None = None,
             ctx=None) -> torch.Tensor:
    """idx: (B, T) prompt ids on the model's device. Returns (B, T + n) ids."""
    device = idx.device
    B, T = idx.shape
    cfg = model.cfg
    total = min(T + max_new_tokens, cfg.max_seq_len)
    cache_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cache = model.make_cache(B, total, device, cache_dtype)
    if ctx is None:
        ctx, _ = autocast_ctx(device)

    with ctx:
        logits, _ = model(idx, cache=cache)          # (B, 1, V)
    out = idx
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(total - T):
        step_logits = logits[:, -1, :].float()
        if repetition_penalty and repetition_penalty != 1.0:
            for b in range(B):
                seen = torch.unique(out[b])
                vals = step_logits[b, seen]
                step_logits[b, seen] = torch.where(
                    vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        step_logits = step_logits / max(temperature, 1e-5)
        step_logits = filter_logits(step_logits, top_k, top_p)
        if temperature <= 1e-5:
            nxt = step_logits.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(step_logits, dim=-1), num_samples=1)
        if eos_id is not None:
            nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, eos_id), nxt)
            done = done | (nxt.squeeze(1) == eos_id)
        out = torch.cat([out, nxt], dim=1)
        if eos_id is not None and bool(done.all()):
            break
        with ctx:
            logits, _ = model(nxt, cache=cache)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(prog="python -m lm.sample")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="data/mix1",
                   help="directory holding tokenizer.json")
    p.add_argument("--prompt", type=str, default="The following is a proof that")
    p.add_argument("--prompt_file", type=str, default="")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)

    from lm.data import EOS_TOKEN, load_tokenizer
    tok = load_tokenizer(args.data_dir)
    eos_id = tok.token_to_id(EOS_TOKEN)

    device = pick_device(args.device)
    model, cfg, ckpt = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt}: step {ckpt['step']}, {ckpt['tokens_seen']:,} tokens, "
          f"{model.num_params()/1e6:.1f}M params")

    text = open(args.prompt_file).read() if args.prompt_file else args.prompt
    ids = tok.encode(text).ids
    assert len(ids) < cfg.max_seq_len, "prompt longer than max_seq_len"
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    idx = idx.repeat(args.num_samples, 1)

    t0 = time.time()
    out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k,
                   args.top_p, args.repetition_penalty, eos_id)
    dt = time.time() - t0
    new = out.size(1) - idx.size(1)
    print(f"--- {new} new tokens in {dt:.2f}s "
          f"({args.num_samples * new / dt:.1f} tok/s total) ---")
    for i in range(out.size(0)):
        print(f"\n===== sample {i} =====")
        print(tok.decode(out[i].tolist()))


if __name__ == "__main__":
    main()