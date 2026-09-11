# Makefile — the single entry point. Every target is one runnable command.
# Override any variable inline:  make train CONFIG=configs/650m.json
# Pass extra flags through:      make train ARGS="--micro_bs 32 --grad_accum 2"

PY           ?= python
CONFIG       ?= configs/300m.json
DATA_DIR     ?= data/mix1
RUN_DIR      ?= $(shell $(PY) -c "import json;print(json.load(open('$(CONFIG)'))['train']['out_dir'])" 2>/dev/null || echo runs/300m)
CKPT         ?= $(RUN_DIR)/ckpt_best.pt
MIXTURE      ?= default
VOCAB        ?= 32768
DOCS         ?= 400000
TRAIN_TOKENS ?= 15e9
VAL_TOKENS   ?= 20e6
TFLOPS       ?= 40
ARGS         ?=

.PHONY: help install smoke smoke-gpu report probes tokenizer pack inspect \
        train resume eval sample tail clean-runs clean-data

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "};{printf "  \033[1m%-11s\033[0m %s\n",$$1,$$2}'

install:  ## install deps + the lm package in editable mode
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

smoke:  ## 27 checks on CPU, ~40s, no downloads — run before any GPU time
	$(PY) tests/smoke_test.py

smoke-gpu:  ## same suite on the GPU, adds the bf16 autocast check
	$(PY) tests/smoke_test.py --device cuda

report:  ## params / memory / GPU-days for every preset
	$(PY) -m lm.config --all --micro_bs 16 --seq_len 2048 --achieved_tflops $(TFLOPS)

probes:  ## regenerate the deterministic math/logic probe sets
	$(PY) tasks/make_probes.py --out_dir tasks --n 200 --seed 0

tokenizer:  ## train the byte-level BPE on a sample of the mixture
	$(PY) -m lm.data tokenizer --out_dir $(DATA_DIR) --mixture $(MIXTURE) \
	  --vocab_size $(VOCAB) --n_docs $(DOCS)

pack:  ## stream + tokenize the corpus into train.bin / val.bin (resumable)
	$(PY) -m lm.data pack --out_dir $(DATA_DIR) --mixture $(MIXTURE) \
	  --train_tokens $(TRAIN_TOKENS) --val_tokens $(VAL_TOKENS)
	$(PY) -m lm.data inspect --out_dir $(DATA_DIR)

inspect:  ## print meta.json and decode a slice of the packed tokens
	$(PY) -m lm.data inspect --out_dir $(DATA_DIR)

train:  ## train from CONFIG
	$(PY) -m lm.train --config $(CONFIG) $(ARGS)

resume:  ## resume the latest checkpoint in that config's out_dir
	$(PY) -m lm.train --config $(CONFIG) --resume latest $(ARGS)

eval:  ## perplexity + every probe set for CKPT, into the run's evals/ dir
	@mkdir -p $(RUN_DIR)/evals
	$(PY) -m lm.evaluate --ckpt $(CKPT) --data_dir $(DATA_DIR) --tasks ppl \
	  --out_json $(RUN_DIR)/evals/ppl.json
	@for f in tasks/*_mc.jsonl; do n=$$(basename $$f .jsonl); \
	  $(PY) -m lm.evaluate --ckpt $(CKPT) --data_dir $(DATA_DIR) --tasks mc \
	    --task_file $$f --out_json $(RUN_DIR)/evals/$$n.json; done
	@for f in tasks/*_gen.jsonl; do n=$$(basename $$f .jsonl); \
	  $(PY) -m lm.evaluate --ckpt $(CKPT) --data_dir $(DATA_DIR) --tasks gen \
	    --task_file $$f --limit 100 --max_new_tokens 48 \
	    --out_json $(RUN_DIR)/evals/$$n.json; done
	@echo "results in $(RUN_DIR)/evals"

sample:  ## generate from CKPT; pass PROMPT="..."
	$(PY) -m lm.sample --ckpt $(CKPT) --data_dir $(DATA_DIR) \
	  $(if $(PROMPT),--prompt "$(PROMPT)",) $(ARGS)

tail:  ## follow the training log
	tail -f $(RUN_DIR)/log.jsonl

clean-runs:  ## delete every run directory (checkpoints included)
	rm -rf runs/* && touch runs/.gitkeep

clean-data:  ## delete packed tokens and tokenizers
	rm -rf data/* && touch data/.gitkeep