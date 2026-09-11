# lm/data.py
"""
Corpus mixture -> BPE tokenizer -> packed uint16 token bins -> batch loader.

HuggingFace is used here and only here: `datasets` for streaming the corpora and
`tokenizers` for BPE training. Everything downstream reads a flat .bin memmap.

Step 1 - train the tokenizer (~20 min on 400k docs):
    python -m lm.data tokenizer --out_dir data/mix1 --vocab_size 32768 --n_docs 400000

Step 2 - pack tokens (long-running; resumable by re-running with a larger
--train_tokens, it appends):
    python -m lm.data pack --out_dir data/mix1 --train_tokens 15e9 --val_tokens 20e6

Inspect:
    python -m lm.data inspect --out_dir data/mix1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

TOKEN_DTYPE = np.uint16  # requires vocab_size < 65536

# Corpus mixture. `token_share` is the share of *tokens* you want from each
# source; `tokens_per_doc` is that corpus's published average, used to convert
# token shares into the document-sampling probabilities interleave_datasets
# actually takes. Without that conversion a 62/22/16 doc split lands at roughly
# 59/30/11 in tokens, because finemath documents are ~40% longer than
# fineweb-edu's and cosmopedia's are ~30% shorter.
MIXTURES: dict[str, list[dict]] = {
    # Balanced default: English fluency from fineweb-edu, math from finemath-4+
    # (9.6B tokens, the subset HF's own ablations show beating OpenWebMath on
    # GSM8K/MATH), and explanatory prose from cosmopedia-v2, which gives a small
    # model clean worked-example structure that raw web text does not.
    "default": [
        dict(path="HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train",
             text_key="text", token_share=0.62, tokens_per_doc=1015),
        dict(path="HuggingFaceTB/finemath", name="finemath-4plus", split="train",
             text_key="text", token_share=0.22, tokens_per_doc=1430),
        dict(path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train",
             text_key="text", token_share=0.16, tokens_per_doc=718),
    ],
    # Math-forward: adds the larger, noisier infiwebmath-3+ pool. Only worth it
    # above ~20B tokens, where 22% of finemath-4+ would start repeating.
    "math_heavy": [
        dict(path="HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train",
             text_key="text", token_share=0.45, tokens_per_doc=1015),
        dict(path="HuggingFaceTB/finemath", name="finemath-4plus", split="train",
             text_key="text", token_share=0.25, tokens_per_doc=1430),
        dict(path="HuggingFaceTB/finemath", name="infiwebmath-3plus", split="train",
             text_key="text", token_share=0.15, tokens_per_doc=1475),
        dict(path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train",
             text_key="text", token_share=0.15, tokens_per_doc=718),
    ],
    # Single small corpus, no gated repos: use this to prove the pipeline works
    # before committing to a multi-day pack.
    "smoke": [
        dict(path="stas/openwebtext-10k", name=None, split="train",
             text_key="text", token_share=1.0, tokens_per_doc=1000),
    ],
}

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SPECIALS = [EOS_TOKEN, PAD_TOKEN]


# --------------------------------------------------------------------------- #
# Streaming corpus
# --------------------------------------------------------------------------- #
def doc_probabilities(specs: list[dict]) -> list[float]:
    """Convert desired token shares into document-sampling probabilities."""
    raw = [s["token_share"] / s["tokens_per_doc"] for s in specs]
    total = sum(raw)
    return [r / total for r in raw]


def source_names(mixture: str) -> list[str]:
    return [f"{s['path']}:{s['name']}" if s["name"] else s["path"]
            for s in MIXTURES[mixture]]


def build_stream(mixture: str, seed: int):
    from datasets import interleave_datasets, load_dataset

    specs = MIXTURES[mixture]
    streams = []
    for i, spec in enumerate(specs):
        ds = load_dataset(spec["path"], name=spec["name"], split=spec["split"],
                          streaming=True)
        cols = list(ds.column_names) if ds.column_names else None
        ds = ds.map(lambda ex, k=spec["text_key"], src=i: {"text": ex[k], "src": src},
                    remove_columns=cols)
        streams.append(ds)
    if len(streams) == 1:
        return streams[0]
    probs = doc_probabilities(specs)
    print("[mixture] " + " | ".join(
        f"{n} token_share={s['token_share']:.2f} doc_prob={p:.3f}"
        for n, s, p in zip(source_names(mixture), specs, probs)))
    # interleave_datasets is dataset plumbing, not modelling: weighted
    # round-robin over the shard iterators, nothing materialized.
    return interleave_datasets(streams, probabilities=probs, seed=seed,
                               stopping_strategy="all_exhausted")


def iter_texts(mixture: str, seed: int, limit: int | None = None):
    """Yields (text, source_index)."""
    stream = build_stream(mixture, seed)
    for i, ex in enumerate(stream):
        if limit is not None and i >= limit:
            return
        text = ex.get("text")
        if text:
            yield text, int(ex.get("src", 0))


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def train_tokenizer(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    # Digits(individual_digits=True) forces every numeral to its own token.
    # Without it BPE merges "1000"/"999" into single units and multi-digit
    # arithmetic has to be memorized per-string instead of computed per-digit.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=2,
    )
    t0 = time.time()
    tok.train_from_iterator((t for t, _ in iter_texts(args.mixture, args.seed,
                                                      args.n_docs)),
                            trainer=trainer, length=args.n_docs)
    path = os.path.join(args.out_dir, "tokenizer.json")
    tok.save(path)
    print(f"[tokenizer] vocab={tok.get_vocab_size()} saved to {path} "
          f"in {time.time() - t0:.0f}s")
    probe = "The derivative of 3x^2 + 17 is 6x. If x = 1024 then f'(x) = 6144."
    ids = tok.encode(probe).ids
    print(f"[tokenizer] probe -> {len(ids)} tokens, roundtrip ok="
          f"{tok.decode(ids).strip() == probe.strip()}")


def load_tokenizer(out_dir: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #
def pack(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    tok = load_tokenizer(args.out_dir)
    vocab_size = tok.get_vocab_size()
    assert vocab_size < 2 ** 16, "uint16 packing requires vocab < 65536"
    eos_id = tok.token_to_id(EOS_TOKEN)
    assert eos_id is not None, f"{EOS_TOKEN} missing from tokenizer"

    train_target = int(float(args.train_tokens))
    val_target = int(float(args.val_tokens))
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    have_train = os.path.getsize(train_path) // 2 if os.path.exists(train_path) else 0
    have_val = os.path.getsize(val_path) // 2 if os.path.exists(val_path) else 0
    print(f"[pack] existing train={have_train:,} val={have_val:,} tokens")
    if have_train >= train_target and have_val >= val_target:
        print("[pack] targets already met, nothing to do")
        return

    f_val = open(val_path, "ab")
    f_train = open(train_path, "ab")
    buf: list[int] = []
    n_val, n_train = have_val, have_train
    n_docs = 0
    t0 = time.time()
    batch: list[str] = []
    batch_src: list[int] = []
    names = source_names(args.mixture)
    src_tokens = [0] * len(names)

    def flush(buffer: list[int]) -> None:
        nonlocal n_val, n_train
        arr = np.asarray(buffer, dtype=TOKEN_DTYPE)
        if n_val < val_target:
            take = min(val_target - n_val, arr.size)
            f_val.write(arr[:take].tobytes())
            n_val += take
            arr = arr[take:]
        if arr.size:
            f_train.write(arr.tobytes())
            n_train += arr.size

    try:
        for text, src in iter_texts(args.mixture, args.seed, None):
            batch.append(text)
            batch_src.append(src)
            n_docs += 1
            if len(batch) < args.encode_batch:
                continue
            for enc, s_i in zip(tok.encode_batch(batch), batch_src):
                buf.extend(enc.ids)
                buf.append(eos_id)
                src_tokens[s_i] += len(enc.ids) + 1
            batch, batch_src = [], []
            if len(buf) >= args.flush_tokens:
                flush(buf)
                buf = []
                elapsed = time.time() - t0
                done = n_train - have_train + n_val - have_val
                print(f"[pack] docs={n_docs:,} train={n_train:,}/{train_target:,} "
                      f"val={n_val:,}/{val_target:,} "
                      f"{done / max(elapsed, 1e-6):,.0f} tok/s", flush=True)
            if n_train >= train_target and n_val >= val_target:
                break
    except KeyboardInterrupt:
        print("[pack] interrupted; flushing buffer")
    finally:
        if batch:
            for enc, s_i in zip(tok.encode_batch(batch), batch_src):
                buf.extend(enc.ids)
                buf.append(eos_id)
                src_tokens[s_i] += len(enc.ids) + 1
        if buf:
            flush(buf)
        f_val.close()
        f_train.close()

    packed = sum(src_tokens)
    realized = {n: round(t / packed, 4) for n, t in zip(names, src_tokens)} \
        if packed else {}
    meta = dict(vocab_size=vocab_size, eos_id=eos_id,
                pad_id=tok.token_to_id(PAD_TOKEN),
                dtype="uint16", mixture=args.mixture, seed=args.seed,
                train_tokens=n_train, val_tokens=n_val, docs_seen=n_docs,
                target_token_share={n: s["token_share"]
                                    for n, s in zip(names, MIXTURES[args.mixture])},
                realized_token_share=realized,
                source_tokens=dict(zip(names, src_tokens)))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pack] {n_train:,} train + {n_val:,} val tokens from {n_docs:,} docs")
    for n, spec in zip(names, MIXTURES[args.mixture]):
        print(f"[pack]   {n}: target {spec['token_share']:.2f} "
              f"realized {realized.get(n, 0):.4f}")


def inspect(args: argparse.Namespace) -> None:
    meta_path = os.path.join(args.out_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    print(json.dumps(meta, indent=2))
    tok = load_tokenizer(args.out_dir)
    for split in ("train", "val"):
        p = os.path.join(args.out_dir, f"{split}.bin")
        if not os.path.exists(p):
            continue
        arr = np.memmap(p, dtype=TOKEN_DTYPE, mode="r")
        print(f"{split}.bin: {arr.size:,} tokens ({arr.size * 2 / 1e9:.2f} GB)")
        print("  sample:", repr(tok.decode(arr[:96].astype(np.int64).tolist()))[:400])


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
class PackedDataset:
    """
    Random fixed-length windows over a flat token memmap. Offsets are drawn from
    a per-step seeded RNG, so a resume at step N reproduces the same batches
    without persisting any loader state.
    """

    def __init__(self, path: str, seq_len: int, batch_size: int, device: str,
                 seed: int = 1337):
        self.path = path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.n_tokens = os.path.getsize(path) // np.dtype(TOKEN_DTYPE).itemsize
        if self.n_tokens < seq_len + 1:
            raise ValueError(f"{path} has {self.n_tokens} tokens, need > {seq_len}")

    def _data(self) -> np.memmap:
        # Re-open per batch: memmap page-cache references otherwise accumulate.
        return np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")

    def batch(self, step: int):
        import torch
        rng = np.random.default_rng(self.seed * 1_000_003 + step)
        data = self._data()
        hi = self.n_tokens - self.seq_len - 1
        ix = rng.integers(0, hi, size=self.batch_size, dtype=np.int64)
        x = np.stack([data[i:i + self.seq_len].astype(np.int64) for i in ix])
        y = np.stack([data[i + 1:i + 1 + self.seq_len].astype(np.int64) for i in ix])
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.startswith("cuda"):
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        else:
            xt, yt = xt.to(self.device), yt.to(self.device)
        return xt, yt

    def sequential_batches(self, batch_size: int | None = None):
        """Non-overlapping windows in order; used by evaluate.py for perplexity."""
        import torch
        bs = batch_size or self.batch_size
        data = self._data()
        n_win = (self.n_tokens - 1) // self.seq_len
        for start in range(0, n_win, bs):
            idxs = [w * self.seq_len for w in range(start, min(start + bs, n_win))]
            x = np.stack([data[i:i + self.seq_len].astype(np.int64) for i in idxs])
            y = np.stack([data[i + 1:i + 1 + self.seq_len].astype(np.int64) for i in idxs])
            yield (torch.from_numpy(x).to(self.device),
                   torch.from_numpy(y).to(self.device))


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(prog="python -m lm.data",
                                description="tokenizer / packing utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out_dir", type=str, default="data/mix1")
    common.add_argument("--mixture", type=str, default="default",
                        choices=sorted(MIXTURES))
    common.add_argument("--seed", type=int, default=1337)

    t = sub.add_parser("tokenizer", parents=[common])
    t.add_argument("--vocab_size", type=int, default=32768)
    t.add_argument("--n_docs", type=int, default=400_000)

    k = sub.add_parser("pack", parents=[common])
    k.add_argument("--train_tokens", type=str, default="15e9")
    k.add_argument("--val_tokens", type=str, default="20e6")
    k.add_argument("--encode_batch", type=int, default=1024)
    k.add_argument("--flush_tokens", type=int, default=4_000_000)

    sub.add_parser("inspect", parents=[common])

    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cmd == "tokenizer":
        train_tokenizer(args)
    elif args.cmd == "pack":
        pack(args)
    else:
        inspect(args)


if __name__ == "__main__":
    main()