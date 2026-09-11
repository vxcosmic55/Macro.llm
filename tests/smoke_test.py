# tests/smoke_test.py
"""
End-to-end verification with no downloads and no GPU required. Run this before
spending a GPU-hour; it exercises every numerical path a long run depends on.

Run:
    python tests/smoke_test.py                 # CPU, fp32, ~40s
    python tests/smoke_test.py --device cuda   # also checks bf16 autocast
    make smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lm.config import (TRAIN_DEFAULTS, ModelConfig, config_from_args,   # noqa: E402
                       memory_report, param_count, resolve_config)
from lm.data import PackedDataset                                       # noqa: E402
from lm.model import Transformer                                        # noqa: E402
from lm.sample import generate                                          # noqa: E402
from lm.utils import numbered_ckpts                                    # noqa: E402

TINY = dict(vocab_size=512, n_layer=3, n_head=4, n_kv_head=2, d_model=128,
            d_ff=256, max_seq_len=128, loss_chunk_tokens=64)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name} {detail}")


def subproc_detail(r: "subprocess.CompletedProcess", label: str) -> str:
    """Prints full raw stdout/stderr (repr'd, so hidden chars are visible)
    on their own lines, then returns a short one-line summary for check().
    Kept verbose deliberately: a truncated/guessed diagnostic is how the
    previous version of this helper hid the very bug it was meant to catch.
    """
    out, err = r.stdout or "", r.stderr or ""
    print(f"  --- {label} raw stdout ({len(out)} chars) ---")
    print(f"  {out!r}")
    print(f"  --- {label} raw stderr ({len(err)} chars) ---")
    print(f"  {err!r}")
    return f"rc={r.returncode} stdout_len={len(out)} stderr_len={len(err)}"


def parse_leading_json(text: str) -> dict:
    """Parses the first JSON value in `text` and ignores anything after it.
    Needed because some Windows terminal-color libraries (colorama, pulled in
    transitively via tqdm) append a trailing ANSI reset code after a script's
    last print, which a strict json.loads() rejects as "Extra data" even
    though the JSON itself is perfectly well-formed.
    """
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--keep", action="store_true", help="keep the temp workdir")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    cfg = ModelConfig(**TINY).validate()
    model = Transformer(cfg).to(device)

    # 1 --------------------------------------------------------------- params
    check("param count matches formula",
          param_count(cfg)["total"] == model.num_params(),
          f"{model.num_params():,}")

    # 2 ------------------------------------------------------- forward shapes
    B, T = 2, 64
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    model.eval()
    with torch.no_grad():
        full, _ = model(idx, all_logits=True)
        last, _ = model(idx)
        _, loss = model(idx, tgt)
    check("logits shapes", tuple(full.shape) == (B, T, cfg.vocab_size)
          and tuple(last.shape) == (B, 1, cfg.vocab_size),
          f"{tuple(full.shape)} / {tuple(last.shape)}")
    check("init loss near ln(V)", abs(loss.item() - np.log(cfg.vocab_size)) < 0.5,
          f"{loss.item():.3f} vs {np.log(cfg.vocab_size):.3f}")

    # 3 ------------------------------------------------------------ causality
    with torch.no_grad():
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 1) % cfg.vocab_size
        full2, _ = model(idx2, all_logits=True)
    drift = (full[:, :-1] - full2[:, :-1]).abs().max().item()
    check("attention is causal", drift < 1e-5, f"max drift {drift:.2e}")

    # 4 -------------------------------------------- chunked loss == unchunked
    with torch.no_grad():
        keep = cfg.loss_chunk_tokens
        model.cfg.loss_chunk_tokens = 0
        _, loss_full = model(idx, tgt)
        model.cfg.loss_chunk_tokens = keep
        _, loss_chunked = model(idx, tgt)
    check("chunked CE == unchunked CE",
          abs(loss_full.item() - loss_chunked.item()) < 1e-5,
          f"{loss_full.item():.6f} vs {loss_chunked.item():.6f}")

    # 5 ------------------------------------- grad checkpointing == plain grads
    model.train()
    grads = {}
    for use_ckpt in (False, True):
        model.zero_grad(set_to_none=True)
        model.enable_gradient_checkpointing(use_ckpt)
        _, l = model(idx, tgt)
        l.backward()
        grads[use_ckpt] = torch.cat(
            [p.grad.detach().reshape(-1).clone() for p in model.parameters()])
    delta = (grads[True] - grads[False]).abs().max().item()
    check("gradient checkpointing matches", delta < 1e-4, f"max |dg| {delta:.2e}")
    check("all params received grads",
          all(p.grad is not None for p in model.parameters()))
    model.zero_grad(set_to_none=True)
    model.enable_gradient_checkpointing(False)

    # 6 --------------------------------------------- KV cache == full forward
    model.eval()
    with torch.no_grad():
        cache = model.make_cache(B, T + 4, device, torch.float32)
        model(idx[:, :-1], cache=cache)
        step_logits, _ = model(idx[:, -1:], cache=cache)
        ref, _ = model(idx, all_logits=True)
    cache_err = (step_logits[:, -1] - ref[:, -1]).abs().max().item()
    check("KV cache == full forward", cache_err < 1e-3,
          f"max err {cache_err:.2e}, pos={cache.pos}")

    # 7 ------------------------------------------------------------- generate
    with torch.no_grad():
        out = generate(model, idx[:, :8], max_new_tokens=16, temperature=0.9,
                       top_k=50, top_p=0.95)
        greedy = generate(model, idx[:, :8], max_new_tokens=16, temperature=0.0)
    check("generate returns prompt + n tokens",
          out.shape == (B, 24) and greedy.shape == (B, 24), f"{tuple(out.shape)}")
    check("generate stays in vocab",
          int(out.max()) < cfg.vocab_size and int(out.min()) >= 0)

    # 8 -------------------------------------------------------- dtype/devices
    devs = {str(p.device) for p in model.parameters()}
    devs |= {str(model.rope_cos.device), str(model.rope_sin.device)}
    check("params and buffers on one device", len(devs) == 1, str(devs))
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l16 = model(idx, tgt)
        check("bf16 autocast forward runs", bool(torch.isfinite(l16).item()),
              f"loss {l16.item():.3f}")

    # 9 ------------------------------------------------------ config layering
    ns = argparse.Namespace(preset=None, d_model=None, n_layer=None, n_head=None,
                            n_kv_head=None, d_ff=None, vocab_size=None,
                            max_seq_len=None, rope_theta=None, dropout=None,
                            loss_chunk_tokens=None, no_tie_embeddings=None)
    from_file = config_from_args(ns, {"preset": "100m", "vocab_size": 1024})
    ns.d_model, ns.preset = 512, "300m"
    cli_wins = config_from_args(ns, {"preset": "100m", "d_model": 4096})
    check("json config drives model config",
          from_file.d_model == 768 and from_file.vocab_size == 1024,
          f"d_model={from_file.d_model} vocab={from_file.vocab_size}")
    check("CLI overrides json config",
          cli_wins.d_model == 512 and cli_wins.n_layer == 24,
          f"d_model={cli_wins.d_model} n_layer={cli_wins.n_layer}")
    layered = resolve_config(TRAIN_DEFAULTS, {"lr": 1e-4, "micro_bs": 8},
                             {"micro_bs": 2})
    check("train config layers defaults < file < cli",
          layered.lr == 1e-4 and layered.micro_bs == 2
          and layered.seq_len == TRAIN_DEFAULTS["seq_len"],
          f"lr={layered.lr} micro_bs={layered.micro_bs}")
    rejected = False
    try:
        resolve_config(TRAIN_DEFAULTS, {"lr_typo": 1}, {})
    except KeyError:
        rejected = True
    check("unknown config keys are rejected", rejected)

    cfg_files = sorted(glob.glob(os.path.join(ROOT, "configs", "*.json")))
    ok_all, bad_file = bool(cfg_files), ""
    for path in cfg_files:
        try:
            blob = json.load(open(path))
            ns2 = argparse.Namespace(**{k: None for k in vars(ns)})
            config_from_args(ns2, blob.get("model", {}))
            resolve_config(TRAIN_DEFAULTS, blob.get("train", {}), {})
        except Exception as e:                                   # noqa: BLE001
            ok_all, bad_file = False, f"{os.path.basename(path)}: {e}"
    check(f"configs/*.json parse and validate ({len(cfg_files)} files)",
          ok_all, bad_file)

    # 10 ------------------------------------------ data loader + train/resume
    work = tempfile.mkdtemp(prefix="lm_smoke_")
    data_dir, run_dir = os.path.join(work, "data"), os.path.join(work, "run")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 200_000), ("val", 20_000)):
        rng.integers(0, cfg.vocab_size, size=n, dtype=np.uint16).tofile(
            os.path.join(data_dir, f"{split}.bin"))
    json.dump({"vocab_size": cfg.vocab_size, "eos_id": 0, "dtype": "uint16"},
              open(os.path.join(data_dir, "meta.json"), "w"))

    ds = PackedDataset(os.path.join(data_dir, "train.bin"), 64, 4, args.device, seed=7)
    x, y = ds.batch(0)
    x2, _ = ds.batch(0)
    x3, _ = ds.batch(1)
    check("loader shapes/dtype/device",
          x.shape == (4, 64) and x.dtype == torch.int64
          and str(x.device).startswith(args.device), f"{tuple(x.shape)} {x.dtype}")
    check("targets are inputs shifted by one", torch.equal(x[:, 1:], y[:, :-1]))
    check("loader deterministic per step, varies across steps",
          torch.equal(x, x2) and not torch.equal(x, x3))

    base = [sys.executable, "-m", "lm.train",
            "--data_dir", data_dir, "--out_dir", run_dir,
            "--preset", "100m", "--vocab_size", str(cfg.vocab_size),
            "--d_model", "128", "--n_layer", "3", "--n_head", "4",
            "--n_kv_head", "2", "--d_ff", "256", "--max_seq_len", "128",
            "--loss_chunk_tokens", "64", "--seq_len", "64", "--micro_bs", "2",
            "--grad_accum", "2", "--device", args.device, "--log_interval", "2",
            "--eval_interval", "4", "--eval_batches", "2", "--ckpt_interval", "4",
            "--lr", "1e-3", "--warmup_tokens", "512"]
    r1 = subprocess.run(base + ["--max_steps", "8"], capture_output=True,
                        text=True, cwd=ROOT)
    check("lm.train runs", r1.returncode == 0,
          (r1.stderr.strip().splitlines() or [""])[-1][:200])
    r2 = subprocess.run(base + ["--max_steps", "16", "--resume", "latest"],
                        capture_output=True, text=True, cwd=ROOT)
    check("lm.train resumes", r2.returncode == 0 and "[resume]" in r2.stdout,
          (r2.stderr.strip().splitlines() or [""])[-1][:200])
    if r1.returncode == 0:
        files = sorted(os.listdir(run_dir))
        check("checkpoints written incl. best and run_config",
              "ckpt_best.pt" in files and "run_config.json" in files
              and len(numbered_ckpts(run_dir)) >= 1, str(files))
        losses = [json.loads(l)["loss"] for l in
                  open(os.path.join(run_dir, "log.jsonl")) if '"loss"' in l]
        check("loss finite and not diverging",
              bool(np.all(np.isfinite(losses))) and losses[-1] <= losses[0] + 0.5,
              f"{losses[0]:.3f} -> {losses[-1]:.3f}")

    # 11 --------------------------------------------- reload + generate + eval
    ckpt_path = os.path.join(run_dir, "ckpt_best.pt")
    r3 = subprocess.run(
        [sys.executable, "-c",
         "import torch;from lm.sample import load_model,generate;"
         f"m,c,k=load_model({ckpt_path!r},torch.device({args.device!r}));"
         f"i=torch.randint(0,c.vocab_size,(1,8),device={args.device!r});"
         "print('OK',tuple(generate(m,i,8,temperature=0.8,top_p=0.95).shape))"],
        capture_output=True, text=True, cwd=ROOT)
    check("checkpoint reloads and generates", r3.returncode == 0 and "OK" in r3.stdout,
          (r3.stdout.strip() or (r3.stderr.strip().splitlines() or [""])[-1])[:200])

    # evaluate.py's main() loads a tokenizer unconditionally, regardless of
    # --tasks, so every lm.evaluate subprocess below needs tokenizer.json to
    # exist first -- build a tiny one offline (tokenizers only, no network).
    tokenizer_ready = False
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        from lm.data import EOS_TOKEN, PAD_TOKEN, SPECIALS
        mini_tok = Tokenizer(models.BPE(unk_token=None))
        mini_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        mini_trainer = trainers.BpeTrainer(
            vocab_size=300, special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=1)
        mini_tok.train_from_iterator(
            ["x", "a", "b", "c", "7", "hello world", "the quick brown fox"] * 20,
            trainer=mini_trainer)
        mini_tok.save(os.path.join(data_dir, "tokenizer.json"))
        tokenizer_ready = mini_tok.token_to_id(EOS_TOKEN) is not None \
            and mini_tok.token_to_id(PAD_TOKEN) is not None
    except ImportError:
        print("  [skip] `tokenizers` not installed; skipping lm.evaluate checks")
    check("tiny offline tokenizer built for lm.evaluate checks", tokenizer_ready)

    if tokenizer_ready:
        r4 = subprocess.run(
            [sys.executable, "-m", "lm.evaluate", "--ckpt", ckpt_path,
             "--data_dir", data_dir, "--tasks", "ppl", "--seq_len", "64",
             "--batch_size", "2", "--max_batches", "3", "--device", args.device],
            capture_output=True, text=True, cwd=ROOT)
        detail4 = subproc_detail(r4, "ppl")
        check("lm.evaluate ppl runs", r4.returncode == 0 and '"ppl"' in r4.stdout,
              detail4)

    # 12 -------------------------------------------------- evaluate.py mc/gen
    mc_path = os.path.join(work, "probe_mc.jsonl")
    with open(mc_path, "w") as f:
        for _ in range(3):
            f.write(json.dumps({"context": "x", "choices": ["a", "b", "c"],
                                "answer": 0}) + "\n")
    if tokenizer_ready:
        r5 = subprocess.run(
            [sys.executable, "-m", "lm.evaluate", "--ckpt", ckpt_path,
             "--data_dir", data_dir, "--tasks", "mc", "--task_file", mc_path,
             "--device", args.device],
            capture_output=True, text=True, cwd=ROOT)
        detail5 = subproc_detail(r5, "mc")
        mc_ok = False
        if r5.returncode == 0:
            try:
                mc = parse_leading_json(r5.stdout)["mc"]
                mc_ok = (mc["n"] == 3 and 0.0 <= mc["acc"] <= 1.0
                        and 0.0 <= mc["acc_len_norm"] <= 1.0)
            except Exception as e:                               # noqa: BLE001
                detail5 = f"{detail5} parse_error={e!r}"
        check("lm.evaluate mc runs and scores every row", mc_ok, detail5)

    gen_path = os.path.join(work, "probe_gen.jsonl")
    with open(gen_path, "w") as f:
        f.write(json.dumps({"prompt": "x", "answer": "7"}) + "\n")
    if tokenizer_ready:
        r6 = subprocess.run(
            [sys.executable, "-m", "lm.evaluate", "--ckpt", ckpt_path,
             "--data_dir", data_dir, "--tasks", "gen", "--task_file", gen_path,
             "--max_new_tokens", "8", "--device", args.device],
            capture_output=True, text=True, cwd=ROOT)
        detail6 = subproc_detail(r6, "gen")
        gen_ok = False
        if r6.returncode == 0:
            try:
                gen = parse_leading_json(r6.stdout)["gen"]
                gen_ok = (gen["n"] == 1 and 0.0 <= gen["exact_match"] <= 1.0
                         and len(gen["examples"]) == 1)
            except Exception as e:                                # noqa: BLE001
                detail6 = f"{detail6} parse_error={e!r}"
        check("lm.evaluate gen runs and decodes every row", gen_ok, detail6)

    # score_continuation directly, in-process: confirms the loss-masking math
    # (context masked, continuation scored) without needing a real tokenizer.
    s1 = n1 = s2 = n2 = None
    tok_stub_ok = False
    try:
        from lm.evaluate import score_continuation
        from lm.sample import load_model as _load_model
        m2, c2, _ = _load_model(ckpt_path, torch.device(args.device))

        class _StubTok:  # avoids needing a real tokenizer.json for this check
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size

            def encode(self, s):
                ids = [min(ord(ch) % self.vocab_size, self.vocab_size - 1)
                       for ch in s]
                return type("E", (), {"ids": ids})()

        stub_tok = _StubTok(c2.vocab_size)
        stub_ctx = torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                  enabled=False)
        s1, n1 = score_continuation(m2, stub_tok, "hello", "world",
                                    torch.device(args.device), stub_ctx)
        s2, n2 = score_continuation(m2, stub_tok, "hello", "",
                                    torch.device(args.device), stub_ctx)
        tok_stub_ok = (n1 == 5 and s1 <= 0 and n2 == 1 and s2 == -1e9)
    except Exception as e:                                        # noqa: BLE001
        print(f"  score_continuation direct check error: {e}")
    check("score_continuation masks context, scores continuation", tok_stub_ok,
          f"s1={s1} n1={n1} s2={s2} n2={n2}")

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"workdir kept at {work}")

    # 13 -------------------------------------------------------- memory sanity
    real = ModelConfig(vocab_size=32768, n_layer=24, n_head=16, n_kv_head=4,
                       d_model=1024, d_ff=2816, max_seq_len=2048)
    mem = memory_report(real, micro_bs=16, seq_len=2048)
    check("300m preset fits 36 GB at micro_bs 16",
          mem["estimated_peak_GB"] < 36.0,
          f"{mem['estimated_peak_GB']:.1f} GB estimated peak")

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()