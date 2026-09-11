# tasks/make_probes.py
"""
Generate small, deterministic probe sets so there is something to evaluate on
from step 1, before any external benchmark is wired up. These are diagnostics,
not benchmarks: they share a template distribution, so a model can look good on
them and still be weak. Treat movement here as a signal to go run ARC/GSM8K.

Run:
    python tasks/make_probes.py --out_dir tasks --n 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random

NONSENSE = ["bloop", "razzie", "lazzie", "wug", "tove", "borogove", "snark",
            "frumious", "vorpal", "mimsy", "gyre", "gimble", "jubjub", "bandersnatch"]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):4d} -> {path}")


def arithmetic(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            a, b = rng.randint(2, 99), rng.randint(2, 30)
            ans = a * b
        elif op == "+":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            ans = a + b
        else:
            a, b = rng.randint(100, 9999), rng.randint(10, 999)
            a, b = max(a, b), min(a, b)
            ans = a - b
        rows.append({"prompt": f"Q: What is {a} {op} {b}?\nA:", "answer": str(ans)})
    return rows


def word_problems(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        per = rng.randint(3, 19)
        boxes = rng.randint(3, 25)
        eaten = rng.randint(1, per * 2)
        total = per * boxes - eaten
        rows.append({
            "prompt": (f"Q: A crate holds {per} apples. There are {boxes} crates. "
                       f"If {eaten} apples are eaten, how many apples remain?\nA:"),
            "answer": str(total)})
    return rows


def syllogisms(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        a, b, c = rng.sample(NONSENSE, 3)
        distractors = [x for x in NONSENSE if x not in (a, b, c)]
        wrong = rng.sample(distractors, 2)
        choices = [f" {c}s", f" {a}s"] + [f" {w}s" for w in wrong]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        rows.append({
            "context": (f"All {a}s are {b}s. All {b}s are {c}s. "
                        f"Therefore, all {a}s are"),
            "choices": shuffled,
            "answer": order.index(0)})
    return rows


def negation(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        a, b = rng.sample(NONSENSE, 2)
        true_first = rng.random() < 0.5
        ctx = (f"No {a}s are {b}s. Kip is a {a}. "
               f"Question: is Kip a {b}?\nAnswer:")
        choices = [" no", " yes"] if true_first else [" yes", " no"]
        rows.append({"context": ctx, "choices": choices,
                     "answer": choices.index(" no")})
    return rows


def sequences(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        start, stepv = rng.randint(1, 20), rng.randint(2, 9)
        kind = rng.choice(["arith", "geom"])
        if kind == "arith":
            seq = [start + i * stepv for i in range(5)]
        else:
            ratio = rng.randint(2, 4)
            seq = [start * ratio ** i for i in range(5)]
        gold = seq[-1]
        body = ", ".join(str(v) for v in seq[:-1])
        wrongs = {gold + stepv, gold - stepv, gold * 2}
        wrongs.discard(gold)
        choices = [f" {gold}"] + [f" {w}" for w in list(wrongs)[:3]]
        order = list(range(len(choices)))
        rng.shuffle(order)
        rows.append({"context": f"{body}, ?\nThe next number is",
                     "choices": [choices[i] for i in order],
                     "answer": order.index(0)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="tasks")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    write_jsonl(os.path.join(args.out_dir, "arith_gen.jsonl"),
                arithmetic(rng, args.n))
    write_jsonl(os.path.join(args.out_dir, "wordprob_gen.jsonl"),
                word_problems(rng, args.n))
    write_jsonl(os.path.join(args.out_dir, "syllogism_mc.jsonl"),
                syllogisms(rng, args.n))
    write_jsonl(os.path.join(args.out_dir, "negation_mc.jsonl"),
                negation(rng, args.n))
    write_jsonl(os.path.join(args.out_dir, "sequence_mc.jsonl"),
                sequences(rng, args.n))
    print("random-choice baseline: 25% (syllogism, sequence), 50% (negation)")


if __name__ == "__main__":
    main()