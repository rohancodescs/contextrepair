#!/usr/bin/env python3
"""
step6_train_t5_rewriter_canard.py

Train a neural conversational query rewriter (T5-small) on CANARD (Wiki-augmented).
Input: History + Question
Output: Rewrite

After training, you will use it in NEWstep4_generate_predictions.py via:
  --mode q4_t5 --t5_rewriter_dir <output_dir> --rewriter_device cpu

Debug run:
  python step6_train_t5_rewriter_canard.py --output_dir ./outputs/t5_rewriter_debug ^
    --max_train_samples 5000 --max_eval_samples 500 --epochs 1

Full run:
  python step6_train_t5_rewriter_canard.py --output_dir ./outputs/t5_rewriter_canard ^
    --epochs 3

Notes:
- On an 8GB GPU, T5-small fine-tunes comfortably with fp16.
- During generation with BART, keep the rewriter on CPU to avoid VRAM contention.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset  # type: ignore
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# metrics (optional but useful for intrinsic reporting)
try:
    import evaluate  # type: ignore
except Exception:
    evaluate = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_source(history: Any, question: str, max_hist: int = 6) -> str:
    """
    history is typically List[str] in this dataset.
    We keep last max_hist utterances for brevity.
    """
    if isinstance(history, list):
        hist = history[-max_hist:]
        hist_str = " <sep> ".join(str(x).strip() for x in hist if str(x).strip())
    else:
        hist_str = str(history).strip()

    question = str(question).strip()
    # T5 style: a task prefix helps
    return f"rewrite: {hist_str} question: {question}" if hist_str else f"rewrite: question: {question}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="gaussalgo/Canard_Wiki-augmented")
    ap.add_argument("--model_name", type=str, default="t5-small")
    ap.add_argument("--output_dir", type=str, default="./outputs/t5_rewriter_canard")

    ap.add_argument("--max_hist", type=int, default=6)
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=64)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--max_train_samples", type=int, default=0, help="If >0, limit train samples (debug).")
    ap.add_argument("--max_eval_samples", type=int, default=0, help="If >0, limit eval samples (debug).")

    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--predict_with_generate", action="store_true", help="Compute metrics using generated rewrites (slower).")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    print("Loading dataset:", args.dataset_name)
    ds = load_dataset(args.dataset_name)
    train_ds = ds["train"]
    eval_ds = ds["test"] if "test" in ds else ds["validation"]

    # Optional debug limits
    if args.max_train_samples > 0:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(eval_ds))))

    print(f"Train rows: {len(train_ds)} | Eval rows: {len(eval_ds)}")

    print("Loading tokenizer/model:", args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        histories = batch["History"]
        questions = batch["Question"]
        rewrites = batch["Rewrite"]

        sources = [format_source(h, q, max_hist=args.max_hist) for h, q in zip(histories, questions)]
        targets = [str(r).strip() for r in rewrites]

        model_inputs = tok(
            sources,
            max_length=args.max_source_len,
            truncation=True,
        )
        labels = tok(
            text_target=targets,
            max_length=args.max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    # Metrics (optional)
    rouge = bleu = None
    if evaluate is not None and args.predict_with_generate:
        try:
            rouge = evaluate.load("rouge")
        except Exception:
            rouge = None
        try:
            bleu = evaluate.load("sacrebleu")
        except Exception:
            bleu = None

    def compute_metrics(eval_pred):
        # Only used if predict_with_generate=True
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Convert to numpy arrays and ensure proper dtype
        preds = np.array(preds)
        labels = np.array(labels)
        
        # Replace -100 in both predictions and labels with pad_token_id
        # This prevents the OverflowError when decoding
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)

        # Decode predictions and labels
        pred_str = tok.batch_decode(preds, skip_special_tokens=True)
        label_str = tok.batch_decode(labels, skip_special_tokens=True)

        pred_str = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        out = {}
        if bleu is not None:
            out["bleu"] = float(bleu.compute(predictions=pred_str, references=[[x] for x in label_str])["score"])
        if rouge is not None:
            r = rouge.compute(predictions=pred_str, references=label_str)
            # keep a compact set
            out["rougeL"] = float(r.get("rougeL", 0.0))
        out["gen_len"] = float(np.mean([len(tok.encode(p, add_special_tokens=False)) for p in pred_str])) if pred_str else 0.0
        return out

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
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
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.max_target_len,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss" if not args.predict_with_generate else "eval_bleu",
        greater_is_better=False if not args.predict_with_generate else True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics if args.predict_with_generate else None,
    )

    print("\nStarting T5 rewriter training...")
    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    print("\nDone. Rewriter saved to:", args.output_dir)
    print("Next: run NEWstep4_generate_predictions.py with --mode q4_t5 and --t5_rewriter_dir", args.output_dir)
    print("Recommendation: use --rewriter_device cpu during BART generation to avoid GPU memory contention.")


if __name__ == "__main__":
    main()