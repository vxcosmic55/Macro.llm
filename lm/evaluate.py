# lm/evaluate.py
"""
Eval harness, written directly against the model: held-out perplexity,
multiple-choice log-likelihood accuracy, and generative exact-match.

Task files are JSONL.
  multiple choice: {"context": "...", "choices": ["a","b"], "answer": 0}
  generative:      {"prompt": "...", "answer": "42"}

Run:
    python -m lm.evaluate --ckpt runs/300m/ckpt_best.pt --data_dir data/mix1 --tasks ppl
    python -m lm.evaluate --ckpt runs/300m/ckpt_best.pt --data_dir data/mix1 \
        --tasks mc --task_file tasks/arc_easy.jsonl --limit 500
    python -m lm.evaluate --ckpt runs/300m/ckpt_best.pt --data_dir data/mix1 \
        --tasks gen --task_file tasks/gsm8k.jsonl --max_new_tokens 256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import numpy as np
import torch

from lm.data import EOS_TOKEN, PackedDataset, load_tokenizer
from lm.sample import generate, load_model
from lm.utils import autocast_ctx, pick_device, set_seed

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


# --------------------------------------------------------------------------- #
def read_jsonl(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


@torch.no_grad()
def eval_ppl(model, data_dir: str, seq_len: int, batch_size: int, device: str,
             max_batches: int, ctx) -> dict:
    ds = PackedDataset(os.path.join(data_dir, "val.bin"), seq_len, batch_size, device)
    total_loss, total_tok = 0.0, 0
    for i, (x, y) in enumerate(ds.sequential_batches()):
        if max_batches and i >= max_batches:
            break
        with ctx:
            _, loss = model(x, y, loss_reduction="sum")
        total_loss += loss.float().item()
        total_tok += y.numel()
    nll = total_loss / max(total_tok, 1)
    return {"tokens": total_tok, "loss": nll, "ppl": math.exp(min(nll, 20))}


@torch.no_grad()
def score_continuation(model, tok, context: str, continuation: str,
                       device: torch.device, ctx) -> tuple[float, int]:
    """Returns (sum log P(continuation | context), n_continuation_tokens)."""
    ctx_ids = tok.encode(context).ids
    cont_ids = tok.encode(continuation).ids
    if not cont_ids:
        return -1e9, 1
    full = ctx_ids + cont_ids
    max_len = model.cfg.max_seq_len
    if len(full) > max_len:                       # left-truncate the context
        cut = len(full) - max_len
        assert cut < len(ctx_ids), "continuation alone exceeds max_seq_len"
        ctx_ids = ctx_ids[cut:]
        full = ctx_ids + cont_ids
    x = torch.tensor(full[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(full[1:], dtype=torch.long, device=device).unsqueeze(0)
    mask_len = max(len(ctx_ids) - 1, 0)  # len(ctx_ids)==0 must mask nothing, not y[:, :-1]
    if mask_len:
        y[:, :mask_len] = -100
    with ctx:
        _, per_tok = model(x, y, loss_reduction="none")   # (T,)
    valid = (y.reshape(-1) != -100)
    return float(-per_tok[valid].float().sum().item()), int(valid.sum().item())


def eval_mc(model, tok, rows: list[dict], device: torch.device, ctx) -> dict:
    n_raw, n_norm = 0, 0
    for r in rows:
        scores = [score_continuation(model, tok, r["context"], c, device, ctx)
                  for c in r["choices"]]
        raw = [s for s, _ in scores]
        norm = [s / max(n, 1) for s, n in scores]
        n_raw += int(int(np.argmax(raw)) == int(r["answer"]))
        n_norm += int(int(np.argmax(norm)) == int(r["answer"]))
    n = max(len(rows), 1)
    return {"n": len(rows), "acc": n_raw / n, "acc_len_norm": n_norm / n}


def extract_number(text: str) -> str | None:
    m = NUM_RE.findall(text.replace("$", ""))
    if not m:
        return None
    s = m[-1].replace(",", "").rstrip(".")
    return s


def eval_gen(model, tok, rows: list[dict], device: torch.device, ctx,
             max_new_tokens: int, stop: str) -> dict:
    eos_id = tok.token_to_id(EOS_TOKEN)
    correct, shown = 0, []
    for r in rows:
        ids = tok.encode(r["prompt"]).ids
        idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = generate(model, idx, max_new_tokens, temperature=0.0,
                       eos_id=eos_id, ctx=ctx)
        gen = tok.decode(out[0, len(ids):].tolist())
        if stop and stop in gen:
            gen = gen.split(stop)[0]
        pred = extract_number(gen)
        gold = extract_number(str(r["answer"])) or str(r["answer"]).strip()
        ok = pred is not None and pred == gold
        correct += int(ok)
        if len(shown) < 3:
            shown.append({"pred": pred, "gold": gold, "gen": gen[:200]})
    n = max(len(rows), 1)
    return {"n": len(rows), "exact_match": correct / n, "examples": shown}


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(prog="python -m lm.evaluate")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="data/mix1")
    p.add_argument("--tasks", type=str, default="ppl",
                   help="comma separated: ppl,mc,gen")
    p.add_argument("--task_file", type=str, default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--stop", type=str, default="\n\n")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    model, cfg, ckpt = load_model(args.ckpt, device)
    ctx, _ = autocast_ctx(device)
    tok = load_tokenizer(args.data_dir)
    seq_len = min(args.seq_len, cfg.max_seq_len)

    results = {"ckpt": args.ckpt, "step": ckpt["step"],
               "tokens_seen": ckpt["tokens_seen"]}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    limit = args.limit or None

    if "ppl" in tasks:
        results["ppl"] = eval_ppl(model, args.data_dir, seq_len, args.batch_size,
                                  str(device), args.max_batches, ctx)
    if "mc" in tasks:
        assert args.task_file, "--task_file required for mc"
        results["mc"] = eval_mc(model, tok, read_jsonl(args.task_file, limit),
                                device, ctx)
    if "gen" in tasks:
        assert args.task_file, "--task_file required for gen"
        results["gen"] = eval_gen(model, tok, read_jsonl(args.task_file, limit),
                                  device, ctx, args.max_new_tokens, args.stop)

    print(json.dumps(results, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()