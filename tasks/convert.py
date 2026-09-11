# tasks/convert.py
"""
Convert public benchmarks into the two JSONL shapes lm/evaluate.py reads.

  multiple choice -> {"context": str, "choices": [str, ...], "answer": int}
  generative      -> {"prompt": str, "answer": str}

Choices keep a leading space: the tokenizer is byte-level, so " Paris" and
"Paris" are different token sequences and only the first continues the context
the way it appears in real text.

Run:
    python tasks/convert.py arc       --config ARC-Easy --out tasks/arc_easy_mc.jsonl
    python tasks/convert.py arc       --config ARC-Challenge --out tasks/arc_chal_mc.jsonl
    python tasks/convert.py hellaswag --out tasks/hellaswag_mc.jsonl --limit 2000
    python tasks/convert.py piqa      --out tasks/piqa_mc.jsonl
    python tasks/convert.py gsm8k     --split test --out tasks/gsm8k_gen.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    kind = "mc" if "choices" in rows[0] else "gen"
    print(f"wrote {len(rows)} {kind} rows -> {path}")


def load(path: str, name: str | None, split: str):
    from datasets import load_dataset
    return load_dataset(path, name=name, split=split)


def conv_arc(args) -> list[dict]:
    rows = []
    for ex in load("allenai/ai2_arc", args.config, args.split):
        labels = list(ex["choices"]["label"])
        if ex["answerKey"] not in labels:
            continue          # a handful of rows carry a key not in the options
        rows.append({"context": ex["question"].strip(),
                     "choices": [" " + t.strip() for t in ex["choices"]["text"]],
                     "answer": labels.index(ex["answerKey"])})
    return rows


def conv_hellaswag(args) -> list[dict]:
    rows = []
    for ex in load("Rowan/hellaswag", None, args.split):
        if ex["label"] == "":
            continue          # the test split is unlabeled
        ctx = (ex["ctx"] or (ex["ctx_a"] + " " + ex["ctx_b"])).strip()
        rows.append({"context": ctx,
                     "choices": [" " + e.strip() for e in ex["endings"]],
                     "answer": int(ex["label"])})
    return rows


def conv_piqa(args) -> list[dict]:
    rows = []
    for ex in load("ybisk/piqa", None, args.split):
        if int(ex["label"]) < 0:
            continue
        rows.append({"context": ex["goal"].strip(),
                     "choices": [" " + ex["sol1"].strip(), " " + ex["sol2"].strip()],
                     "answer": int(ex["label"])})
    return rows


def conv_gsm8k(args) -> list[dict]:
    rows = []
    for ex in load("openai/gsm8k", "main", args.split):
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        if not re.fullmatch(r"-?\d+(\.\d+)?", gold):
            continue
        rows.append({"prompt": f"Q: {ex['question'].strip()}\nA:", "answer": gold})
    return rows


CONVERTERS = {"arc": conv_arc, "hellaswag": conv_hellaswag,
              "piqa": conv_piqa, "gsm8k": conv_gsm8k}
DEFAULT_SPLIT = {"arc": "validation", "hellaswag": "validation",
                 "piqa": "validation", "gsm8k": "test"}


def main() -> None:
    p = argparse.ArgumentParser(prog="python tasks/convert.py")
    p.add_argument("task", choices=sorted(CONVERTERS))
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--config", type=str, default="ARC-Easy",
                   help="ARC subset: ARC-Easy or ARC-Challenge")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    args.split = args.split or DEFAULT_SPLIT[args.task]

    rows = CONVERTERS[args.task](args)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"{args.task}/{args.split} produced no usable rows")

    for r in rows:                      # fail loudly rather than at eval time
        if "choices" in r:
            assert 0 <= r["answer"] < len(r["choices"]), r
            assert len(r["choices"]) >= 2, r
        else:
            assert r["answer"] and r["prompt"], r
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()