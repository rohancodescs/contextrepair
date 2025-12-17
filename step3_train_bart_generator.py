#!/usr/bin/env python3
"""
step3_train_bart_generator.py

Train BART-base generator (Task II) using:
- examples_train.jsonl / examples_val.jsonl
- retrieved_train_q2_k{K}.jsonl (precomputed BM25 top-k for train, Q2 baseline)
- For val (and any missing train ids), BM25 fallback retrieval using Q2 concat

Designed for my GPU (RTX 4060 Ti with 8GB VRAM):
- fp16
- batch_size=1 default
- gradient accumulation
- optional gradient checkpointing
- per-passage truncation to control input length with k=5

How to run (ensure functionality) : 
  python step3_train_bart_generator.py --data_dir ./data --k 5 ^
    --limit_train 2000 --limit_val 500 --epochs 1 ^
    --output_dir ./outputs/bart_debug ^
    --batch_size 1 --grad_accum 8 --max_source_len 512 --max_passage_tokens 80 ^
    --gradient_checkpointing

Run for FULL CACHEING (did not achieve due to limited system resources):
  python step3_train_bart_generator.py --data_dir ./data --k 5 ^
    --epochs 3 --output_dir ./outputs/bart_full ^
    --batch_size 1 --grad_accum 8 --max_source_len 512 --max_passage_tokens 80 ^
    --gradient_checkpointing
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from rank_bm25 import BM25Okapi 
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from tqdm import tqdm  
except Exception:
    tqdm = None

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# IO helper func
# Ensures the provided directory exists.
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Streams JSON objects from a newline-delimited JSON file.
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# Loads JSONL rows into memory, optionally truncating for debugging.
def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(iter_jsonl(path)):
        out.append(row)
        if limit is not None and (i + 1) >= limit:
            break
    return out


# Applies the regex tokenizer used for BM25 scoring.
def tokenize_bm25(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


# Returns the top-k (index, score) pairs from a BM25 score vector.
def topk_indices_and_scores(scores: Any, k: int) -> List[Tuple[int, float]]:
    """
    Return [(idx, score), ...] sorted desc, length k.
    Uses numpy if available, else pure python.
    """
    try:
        import numpy as np  

        if isinstance(scores, np.ndarray):
            k_eff = min(k, scores.shape[0])
            if k_eff <= 0:
                return []
            idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
            idx = idx[np.argsort(-scores[idx])]
            return [(int(i), float(scores[int(i)])) for i in idx]
    except Exception:
        pass

    n = len(scores)
    k_eff = min(k, n)
    ranked = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)[:k_eff]
    return [(i, float(scores[i])) for i in ranked]


# building queries
# Builds the Q2 concat retrieval query from history and user input.
def build_query_q2_concat(history: List[Dict[str, Any]], user_turn: str, max_turns_concat: int = 6) -> str:
    parts: List[str] = []
    hist_tail = history[-max_turns_concat:] if max_turns_concat > 0 else history
    for t in hist_tail:
        role = str(t.get("role", "")).strip().lower()
        utt = str(t.get("utterance", "")).strip()
        if utt:
            parts.append(f"{role}: {utt}")
    parts.append(f"user: {user_turn.strip()}")
    return " ".join(parts)


# passage loading
# bm25 loading
@dataclass
class PassageStore:
    passage_ids: List[str]
    passage_texts: List[str]
    passage_id_to_text: Dict[str, str]
    bm25: BM25Okapi


# Loads passages, tokenizes them, and constructs a BM25 store.
def load_passages_and_build_bm25(passages_jsonl: Path) -> PassageStore:
    passage_ids: List[str] = []
    passage_texts: List[str] = []
    for row in iter_jsonl(passages_jsonl):
        pid = str(row["passage_id"])
        txt = str(row["text"])
        passage_ids.append(pid)
        passage_texts.append(txt)

    if not passage_ids:
        raise ValueError(f"Loaded 0 passages from {passages_jsonl}")

    tokenized = [tokenize_bm25(t) for t in passage_texts]
    bm25 = BM25Okapi(tokenized)
    pid_to_text = {pid: txt for pid, txt in zip(passage_ids, passage_texts)}
    return PassageStore(passage_ids=passage_ids, passage_texts=passage_texts, passage_id_to_text=pid_to_text, bm25=bm25)


# Retrieves top passage ids for a query using the BM25 store.
def bm25_retrieve_pids(
    store: PassageStore,
    query: str,
    k: int,
) -> List[str]:
    scores = store.bm25.get_scores(tokenize_bm25(query))
    top = topk_indices_and_scores(scores, k)
    return [store.passage_ids[i] for i, _ in top]


# retrieval cache loader (training)
# Loads the precomputed BM25 cache mapping example ids to passage ids.
def load_train_retrieval_cache(cache_jsonl: Path) -> Dict[str, List[str]]:
    """
    Map example_id -> list of passage_ids (ordered).
    """
    m: Dict[str, List[str]] = {}
    if not cache_jsonl.exists():
        return m
    for row in iter_jsonl(cache_jsonl):
        ex_id = str(row.get("example_id", ""))
        retrieved = row.get("retrieved", [])
        if not ex_id or not isinstance(retrieved, list):
            continue
        pids: List[str] = []
        for r in retrieved:
            if isinstance(r, dict) and r.get("passage_id") is not None:
                pids.append(str(r["passage_id"]))
        if pids:
            m[ex_id] = pids
    return m


# format for input
# Truncates text using a tokenizer-level token budget.
def truncate_text_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    """
    Truncate raw text to at most max_tokens (tokenizer tokens).
    """
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


# Builds the formatted seq2seq prompt including history and evidence.
def format_model_input(
    tokenizer: Any,
    history: List[Dict[str, Any]],
    user_turn: str,
    evidence_texts: List[str],
    max_passage_tokens: int,
) -> str:
    # blocks for history
    hist_lines: List[str] = []
    for t in history:
        role = str(t.get("role", "")).strip().lower()
        utt = str(t.get("utterance", "")).strip()
        if not utt:
            continue
        role_label = "User" if role == "user" else "Agent"
        # keep history pretty short 
        utt = re.sub(r"\s+", " ", utt)
        if len(utt) > 240:
            utt = utt[:237] + "..."
        hist_lines.append(f"{role_label}: {utt}")

    history_block = "\n".join(hist_lines) if hist_lines else "(none)"

    # evidence block
    ev_lines: List[str] = []
    for i, ptxt in enumerate(evidence_texts[:], start=1):
        ptxt = re.sub(r"\s+", " ", ptxt).strip()
        ptxt = truncate_text_to_tokens(tokenizer, ptxt, max_tokens=max_passage_tokens)
        ev_lines.append(f"[{i}] {ptxt}")

    evidence_block = "\n".join(ev_lines) if ev_lines else "(none)"

    prompt = (
        "HISTORY:\n"
        f"{history_block}\n\n"
        "QUESTION:\n"
        f"{user_turn.strip()}\n\n"
        "EVIDENCE:\n"
        f"{evidence_block}\n\n"
        "INSTRUCTION:\n"
        "Write the next agent response. Use the evidence when relevant.\n"
    )
    return prompt


# dataset
class GenDataset(torch.utils.data.Dataset):
    # Initializes dataset storage and attaches retrieval evidence.
    def __init__(
        self,
        *,
        examples: List[Dict[str, Any]],
        store: PassageStore,
        train_cache: Dict[str, List[str]],
        tokenizer: Any,
        k: int,
        max_turns_concat: int,
        max_source_len: int,
        max_target_len: int,
        max_passage_tokens: int,
        require_cache: bool,
        split_name: str,
    ) -> None:
        self.examples = examples
        self.store = store
        self.train_cache = train_cache
        self.tokenizer = tokenizer
        self.k = k
        self.max_turns_concat = max_turns_concat
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.max_passage_tokens = max_passage_tokens
        self.require_cache = require_cache
        self.split_name = split_name

        # attach evidence pids to skip over the retrieval process inside __getitem__
        self._evidence_pids: List[List[str]] = []
        self._attach_evidence()

    # Attaches BM25 evidence passage ids to each example at init time.
    def _attach_evidence(self) -> None:
        use_tqdm = tqdm is not None
        it = tqdm(self.examples, desc=f"Attach evidence ({self.split_name})") if use_tqdm else self.examples

        missing = 0
        for ex in it:  
            ex_id = str(ex["example_id"])
            history = ex["history"]
            user_turn = str(ex["user_turn"])

            pids = self.train_cache.get(ex_id)
            if pids is None:
                if self.require_cache and self.split_name == "train":
                    missing += 1
                    self._evidence_pids.append([])
                    continue
                # fallback for BM25 retrieval
                q = build_query_q2_concat(history, user_turn, max_turns_concat=self.max_turns_concat)
                pids = bm25_retrieve_pids(self.store, q, k=self.k)

            self._evidence_pids.append(list(pids)[: self.k])

        if missing > 0:
            print(f"ISSUE MAYBE {missing} train examples missing retrieval cache and require_cache=True. They will have empty evidence.")

    # Returns the number of examples.
    def __len__(self) -> int:
        return len(self.examples)

    # Tokenizes a single example into seq2seq inputs and labels.
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        history = ex["history"]
        user_turn = str(ex["user_turn"])
        target = str(ex["target"])

        pids = self._evidence_pids[idx]
        evidence_texts = [self.store.passage_id_to_text.get(pid, "") for pid in pids if pid in self.store.passage_id_to_text]
        src_text = format_model_input(
            self.tokenizer, history, user_turn, evidence_texts, max_passage_tokens=self.max_passage_tokens
        )

        model_inputs = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_source_len,
            padding=False,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                truncation=True,
                max_length=self.max_target_len,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# seeding
# Sets random seeds for python, numpy, and torch.
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np 

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Parses CLI args, prepares data, and launches BART training.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--output_dir", type=str, default="outputs/bart_run")
    ap.add_argument("--model_name", type=str, default="facebook/bart-base")

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_turns_concat", type=int, default=6)

    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=128)
    ap.add_argument("--max_passage_tokens", type=int, default=80)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)

    ap.add_argument("--require_train_cache", action="store_true", help="If set, missing train cache => empty evidence")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--no_fp16", action="store_true")

    ap.add_argument("--logging_steps", type=int, default=50)
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    processed_dir = data_dir / "processed"

    examples_train_path = processed_dir / "examples_train.jsonl"
    examples_val_path = processed_dir / "examples_val.jsonl"
    passages_path = processed_dir / "passages.jsonl"
    cache_train_path = processed_dir / f"retrieved_train_q2_k{args.k}.jsonl"

    for p in [examples_train_path, examples_val_path, passages_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}. make sure you run step1 and step2")

    print("Loading passages + building BM25...")
    store = load_passages_and_build_bm25(passages_path)
    print(f"Passages: {len(store.passage_ids)}")

    print("Loading train retrieval cache:", cache_train_path)
    train_cache = load_train_retrieval_cache(cache_train_path)
    print(f"Train cache rows: {len(train_cache)}")

    # load the examples
    limit_train = args.limit_train if args.limit_train > 0 else None
    limit_val = args.limit_val if args.limit_val > 0 else None

    print("Loading train examples:", examples_train_path)
    train_examples = load_jsonl(examples_train_path, limit=limit_train)
    print("Loading val examples:", examples_val_path)
    val_examples = load_jsonl(examples_val_path, limit=limit_val)
    print(f"Examples loaded: train={len(train_examples)} val={len(val_examples)}")

    # tokenizer
    # model
    print("Loading tokenizer/model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # datasets
    train_ds = GenDataset(
        examples=train_examples,
        store=store,
        train_cache=train_cache,
        tokenizer=tokenizer,
        k=args.k,
        max_turns_concat=args.max_turns_concat,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        max_passage_tokens=args.max_passage_tokens,
        require_cache=args.require_train_cache,
        split_name="train",
    )
    val_ds = GenDataset(
        examples=val_examples,
        store=store,
        train_cache={},  
        tokenizer=tokenizer,
        k=args.k,
        max_turns_concat=args.max_turns_concat,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        max_passage_tokens=args.max_passage_tokens,
        require_cache=False,
        split_name="val",
    )

    # collator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    # train the args
    safe_mkdir(Path(args.output_dir))
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=(not args.no_fp16),
        report_to="none",
        dataloader_num_workers=0,
        predict_with_generate=False,  
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )

    # trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # training using above^^^
    print("\nStarting training...")
    trainer.train()

    # save final model
    print("\nSaving final model + tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # save path for config snapshot
    cfg_path = Path(args.output_dir) / "run_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print("Saved:", cfg_path)


if __name__ == "__main__":
    main()
