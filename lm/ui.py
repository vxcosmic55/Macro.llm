# lm/ui.py
"""
Simple local web UI for sampling from a checkpoint. Wraps lm.sample.generate()
in a Gradio interface: prompt box, sampling controls, and a checkpoint picker
that scans a run directory for every ckpt_*.pt / ckpt_best.pt found.

Not a claude.ai artifact -- this runs wherever the checkpoint lives (the GPU
box during training, or wherever you copy checkpoints to afterward), since it
needs direct access to load the model.

Install (not in requirements.txt -- optional, UI-only):
    pip install gradio

Run:
    python -m lm.ui --data_dir data/mix1 --runs_dir runs/300m
    python -m lm.ui --data_dir data/mix1 --runs_dir runs/300m --share
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os

import torch

from lm.data import EOS_TOKEN, load_tokenizer
from lm.sample import generate, load_model
from lm.utils import autocast_ctx, pick_device

# Mutable module-level cache so the model is loaded once and reused across
# generate() calls, and reloaded only when the checkpoint selection changes --
# reloading a 300M model on every single click would make the UI unusable.
_STATE = {"path": None, "model": None, "cfg": None}


def list_checkpoints(runs_dir: str) -> list[str]:
    if not os.path.isdir(runs_dir):
        return []
    paths = sorted(glob.glob(os.path.join(runs_dir, "ckpt_*.pt")))
    # ckpt_best.pt first, since it's the one most people want by default.
    best = os.path.join(runs_dir, "ckpt_best.pt")
    paths = [p for p in paths if os.path.basename(p) != "ckpt_best.pt"]
    return ([best] if os.path.exists(best) else []) + sorted(paths, reverse=True)


def get_model(path: str, device: torch.device):
    if _STATE["path"] != path:
        if _STATE["model"] is not None:
            del _STATE["model"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        model, cfg, ckpt = load_model(path, device)
        _STATE.update(path=path, model=model, cfg=cfg)
        print(f"[ui] loaded {path}: step {ckpt['step']}, "
              f"{ckpt['tokens_seen']:,} tokens, {model.num_params()/1e6:.1f}M params")
    return _STATE["model"], _STATE["cfg"]


def build_app(args: argparse.Namespace):
    import gradio as gr

    device = pick_device(args.device)
    tok = load_tokenizer(args.data_dir)
    eos_id = tok.token_to_id(EOS_TOKEN)
    ctx, _ = autocast_ctx(device)

    def refresh_choices():
        ck = list_checkpoints(args.runs_dir)
        if not ck:
            return gr.update(choices=[], value=None)
        return gr.update(choices=ck, value=ck[0])

    def run_generate(ckpt_path, prompt, max_new_tokens, temperature, top_k,
                     top_p, repetition_penalty, seed):
        if not ckpt_path:
            return "No checkpoint selected -- click Refresh, or check --runs_dir."
        if not prompt.strip():
            return "Enter a prompt first."
        torch.manual_seed(int(seed))
        model, cfg = get_model(ckpt_path, device)
        model.eval()
        ids = tok.encode(prompt).ids
        if len(ids) >= cfg.max_seq_len:
            return (f"Prompt is {len(ids)} tokens; this model's max_seq_len "
                    f"is {cfg.max_seq_len}. Shorten the prompt.")
        idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = generate(model, idx, int(max_new_tokens), float(temperature),
                           int(top_k), float(top_p), float(repetition_penalty),
                           eos_id, ctx)
        return tok.decode(out[0].tolist())

    with gr.Blocks(title="scratch-lm sampler") as app:
        gr.Markdown("## scratch-lm checkpoint sampler\n"
                    "Pure next-token completion -- no instruction tuning. "
                    "It continues your prompt, it doesn't answer it.")
        with gr.Row():
            ckpt_dd = gr.Dropdown(label="Checkpoint",
                                  choices=list_checkpoints(args.runs_dir),
                                  value=(list_checkpoints(args.runs_dir) or [None])[0])
            refresh_btn = gr.Button("Refresh", scale=0)
        prompt_box = gr.Textbox(label="Prompt", lines=4,
                                placeholder="Once upon a time,")
        with gr.Row():
            max_new = gr.Slider(1, 512, value=128, step=1, label="Max new tokens")
            temp = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
        with gr.Row():
            top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k (0 = off)")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
        with gr.Row():
            rep_pen = gr.Slider(1.0, 2.0, value=1.0, step=0.05,
                                label="Repetition penalty (1.0 = off)")
            seed = gr.Number(value=1337, precision=0, label="Seed")
        gen_btn = gr.Button("Generate", variant="primary")
        output = gr.Textbox(label="Output", lines=10)

        refresh_btn.click(refresh_choices, outputs=ckpt_dd)
        gen_btn.click(run_generate,
                     inputs=[ckpt_dd, prompt_box, max_new, temp, top_k, top_p,
                             rep_pen, seed],
                     outputs=output)
        prompt_box.submit(run_generate,
                          inputs=[ckpt_dd, prompt_box, max_new, temp, top_k,
                                  top_p, rep_pen, seed],
                          outputs=output)

    return app


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m lm.ui")
    p.add_argument("--data_dir", type=str, default="data/mix1",
                   help="directory holding tokenizer.json")
    p.add_argument("--runs_dir", type=str, default="runs/300m",
                   help="directory to scan for ckpt_*.pt files")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true",
                   help="create a public gradio.live link (e.g. to view from "
                        "your laptop while the UI runs on the GPU box)")
    args = p.parse_args()

    if importlib.util.find_spec("gradio") is None:
        raise SystemExit("Gradio isn't installed. Run: pip install gradio")

    app = build_app(args)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()