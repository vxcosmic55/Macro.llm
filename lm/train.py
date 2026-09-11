# lm/train.py
"""
Pretraining loop: bf16 autocast, gradient checkpointing, gradient accumulation,
AdamW with warmup + cosine decay, periodic val, atomic checkpointing, resume.

Config layering: TRAIN_DEFAULTS < --config file < explicit CLI flags.

Run:
    python -m lm.train --config configs/300m.json
    python -m lm.train --config configs/300m.json --micro_bs 32 --grad_accum 2
    python -m lm.train --config configs/300m.json --resume latest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from lm.config import (TRAIN_DEFAULTS, ModelConfig, add_model_args,
                       config_from_args, flops_per_token, load_json_config,
                       memory_report, param_count, resolve_config, token_budget)
from lm.data import PackedDataset
from lm.model import Transformer
from lm.utils import (JsonlLogger, autocast_ctx, enable_tf32, latest_ckpt,
                      load_ckpt, pick_device, prune_ckpts, restore_rng,
                      save_ckpt, set_seed)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m lm.train")
    add_model_args(p)
    p.add_argument("--config", type=str, default=None,
                   help="json run config with optional 'model' and 'train' sections")
    p.add_argument("--resume", type=str, default=None,
                   help="'latest' or a path to a ckpt_*.pt")

    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--micro_bs", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--total_tokens", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min_lr_frac", type=float, default=None)
    p.add_argument("--warmup_tokens", type=str, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)

    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--eval_batches", type=int, default=None)
    p.add_argument("--ckpt_interval", type=int, default=None)
    p.add_argument("--keep_last", type=int, default=None)
    p.add_argument("--achieved_tflops", type=float, default=None)

    p.add_argument("--no_grad_checkpointing", action="store_true", default=None)
    p.add_argument("--compile", action="store_true", default=None)
    return p.parse_args()


def train_cli_overrides(args: argparse.Namespace) -> dict:
    cli = {k: v for k, v in vars(args).items()
           if k in TRAIN_DEFAULTS and v is not None}
    if getattr(args, "no_grad_checkpointing", None):
        cli["grad_checkpointing"] = False
    return cli


def lr_at(step: int, warmup_steps: int, max_steps: int,
          lr: float, min_lr_frac: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return lr * min_lr_frac
    prog = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, dataset: PackedDataset, n_batches: int, ctx) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dataset.sequential_batches()):
        if i >= n_batches:
            break
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.float().item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    file_cfg = load_json_config(args.config)
    run = resolve_config(TRAIN_DEFAULTS, file_cfg.get("train", {}),
                         train_cli_overrides(args))

    os.makedirs(run.out_dir, exist_ok=True)
    set_seed(run.seed)
    enable_tf32()
    device = pick_device(run.device)
    ctx, bf16_on = autocast_ctx(device)
    if device.type == "cuda" and not bf16_on:
        print("[warn] bf16 unsupported here; running fp32 (expect ~2x slower)")

    # ---- model config: a resume always wins, so shapes can never drift ----- #
    ckpt = None
    if args.resume:
        path = latest_ckpt(run.out_dir) if args.resume == "latest" else args.resume
        ckpt = load_ckpt(path)
        cfg = ModelConfig.from_dict(ckpt["model_config"])
        print(f"[resume] {path} at step {ckpt['step']}, {ckpt['tokens_seen']:,} tokens")
    else:
        cfg = config_from_args(args, file_cfg.get("model", {}))
        meta_path = os.path.join(run.data_dir, "meta.json")
        if os.path.exists(meta_path) and args.vocab_size is None:
            meta = json.load(open(meta_path))
            if meta.get("vocab_size"):
                cfg.vocab_size = int(meta["vocab_size"])
        cfg.max_seq_len = max(cfg.max_seq_len, run.seq_len)
        cfg.validate()
    assert run.seq_len <= cfg.max_seq_len, (
        f"seq_len {run.seq_len} > model max_seq_len {cfg.max_seq_len}")

    # ---- data -------------------------------------------------------------- #
    train_ds = PackedDataset(os.path.join(run.data_dir, "train.bin"), run.seq_len,
                             run.micro_bs, str(device), seed=run.seed)
    val_ds = PackedDataset(os.path.join(run.data_dir, "val.bin"), run.seq_len,
                           run.micro_bs, str(device), seed=run.seed + 1)

    tokens_per_step = run.micro_bs * run.grad_accum * run.seq_len
    total_tokens = int(float(run.total_tokens))
    max_steps = run.max_steps or max(total_tokens // tokens_per_step, 1)
    warmup_steps = max(int(float(run.warmup_tokens)) // tokens_per_step, 1)

    # ---- model ------------------------------------------------------------- #
    model = Transformer(cfg).to(device)
    model.enable_gradient_checkpointing(bool(run.grad_checkpointing))
    optimizer, n_decay, n_nodecay = model.configure_optimizers(
        run.weight_decay, run.lr, (run.beta1, run.beta2), device.type)

    step, tokens_seen, best_val = 0, 0, float("inf")
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        best_val = ckpt.get("best_val", float("inf"))
        restore_rng(ckpt)
        prev = ckpt.get("run_config", {})
        prev_tps = (prev.get("micro_bs", 0) * prev.get("grad_accum", 0)
                    * prev.get("seq_len", 0))
        if prev_tps and prev_tps != tokens_per_step:
            print(f"[warn] tokens/step changed {prev_tps:,} -> {tokens_per_step:,}; "
                  f"the LR schedule is re-derived from the new value")
        del ckpt

    raw_model = model
    if run.compile:
        # torch >= 2.1 to coexist with activation checkpointing.
        model = torch.compile(model)

    # ---- reports ----------------------------------------------------------- #
    pc = param_count(cfg)
    mem = memory_report(cfg, run.micro_bs, run.seq_len,
                        grad_checkpointing=bool(run.grad_checkpointing))
    tb = token_budget(cfg, total_tokens, run.seq_len, run.achieved_tflops)
    print(json.dumps(cfg.to_dict(), indent=2))
    print(f"params total {pc['total']/1e6:.1f}M | non-emb {pc['non_embedding']/1e6:.1f}M "
          f"| decay {n_decay/1e6:.1f}M | no-decay {n_nodecay/1e3:.1f}K")
    print(f"effective batch: {run.micro_bs}x{run.grad_accum} = "
          f"{run.micro_bs*run.grad_accum} seqs = {tokens_per_step:,} tokens/step")
    print(f"schedule: {max_steps:,} steps, warmup {warmup_steps:,}, "
          f"{total_tokens/1e9:.2f}B tokens, {tb['tokens_per_param']:.1f} tok/param")
    print(f"memory estimate: {mem['estimated_peak_GB']:.1f} GB peak "
          f"(weights+grads+opt {mem['params_GB']+mem['grads_GB']+mem['optimizer_GB']:.1f}"
          f", acts {mem['activations_GB']:.1f}, loss head {mem['loss_head_GB']:.1f})")
    print(f"train tokens on disk: {train_ds.n_tokens:,} "
          f"({total_tokens/max(train_ds.n_tokens,1):.2f} epochs)")
    with open(os.path.join(run.out_dir, "run_config.json"), "w") as f:
        json.dump({"model": cfg.to_dict(), "train": vars(run)}, f, indent=2)

    logger = JsonlLogger(os.path.join(run.out_dir, "log.jsonl"))
    fpt = flops_per_token(cfg, run.seq_len)
    run_dict = vars(run)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    while step < max_steps:
        lr = lr_at(step, warmup_steps, max_steps, run.lr, run.min_lr_frac)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_acc = 0.0
        for micro in range(run.grad_accum):
            x, y = train_ds.batch(step * run.grad_accum + micro)
            with ctx:
                _, loss = model(x, y)
            loss_acc += loss.detach().float().item() / run.grad_accum
            (loss / run.grad_accum).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tokens_per_step

        if step % run.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            tps = run.log_interval * tokens_per_step / max(dt, 1e-9)
            logger.write(dict(
                step=step, loss=round(loss_acc, 4), lr=round(lr, 8),
                grad_norm=round(float(grad_norm), 3), tokens=tokens_seen,
                tok_per_s=round(tps), tflops=round(tps * fpt / 1e12, 1),
                mem_GB=(round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                        if device.type == "cuda" else 0.0)))

        if step % run.eval_interval == 0 or step == max_steps:
            val = evaluate(model, val_ds, run.eval_batches, ctx)
            logger.write(dict(step=step, val_loss=round(val, 4),
                              val_ppl=round(math.exp(min(val, 20)), 2)))
            if val < best_val:
                best_val = val
                save_ckpt(os.path.join(run.out_dir, "ckpt_best.pt"), raw_model,
                          optimizer, cfg.to_dict(), step, tokens_seen,
                          run_dict, best_val)
            t0 = time.time()

        if step % run.ckpt_interval == 0 or step == max_steps:
            save_ckpt(os.path.join(run.out_dir, f"ckpt_{step:08d}.pt"), raw_model,
                      optimizer, cfg.to_dict(), step, tokens_seen,
                      run_dict, best_val)
            prune_ckpts(run.out_dir, run.keep_last)
            t0 = time.time()

    logger.close()
    print(f"done: {step} steps, {tokens_seen:,} tokens, best val {best_val:.4f}")


if __name__ == "__main__":
    main()