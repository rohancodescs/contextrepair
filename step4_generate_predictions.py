#!/usr/bin/env python3
"""
step4_generate_predictions.py

Generate predictions for a split (val/test/train) under a retrieval query mode:
- q1: user turn only
- q2: concat history + user turn (baseline)

This uses BM25 retrieval over passages.jsonl (built in step1).

Run (debug first):
  python step4_generate_predictions.py --data_dir ./data --model_dir ./outputs/bart_debug ^
    --split val --mode q2 --k 5 --limit 200 --batch_size 1

Then full val:
  python step4_generate_predictions.py --data_dir ./data --model_dir ./outputs/bart_debug ^
    --split val --mode q2 --k 5 --batch_size 1

Then compare q1 vs q2:
  python step4_generate_predictions.py --data_dir ./data --model_dir ./outputs/bart_debug ^
    --split val --mode q1 --k 5 --batch_size 1
  python step4_generate_predictions.py --data_dir ./data --model_dir ./outputs/bart_debug ^
    --split val --mode q2 --k 5 --batch_size 1

Outputs:
  outputs/predictions/{model_name}/{split}_{mode}_k{K}.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from rank_bm25 import BM25Okapi  # type: ignore
from transformers import AutoTokenizer, BartForConditionalGeneration

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# -------------------------
# IO helpers
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(iter_jsonl(path)):
        out.append(row)
        if limit is not None and (i + 1) >= limit:
            break
    return out


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    safe_mkdir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# BM25 helpers
# -------------------------
def tokenize_bm25(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def topk_indices_and_scores(scores: Any, k: int) -> List[Tuple[int, float]]:
    """
    Return [(idx, score), ...] sorted desc, length k.
    Uses numpy if available; falls back to pure python.
    """
    try:
        import numpy as np  # type: ignore

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


def load_passages(passages_jsonl: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (passage_ids, doc_ids, texts)
    """
    pids: List[str] = []
    doc_ids: List[str] = []
    texts: List[str] = []
    for row in iter_jsonl(passages_jsonl):
        pids.append(str(row["passage_id"]))
        doc_ids.append(str(row["doc_id"]))
        texts.append(str(row["text"]))
    if not pids:
        raise ValueError(f"Loaded 0 passages from {passages_jsonl}")
    return pids, doc_ids, texts


# -------------------------
# Query modes
# -------------------------
def build_query_q1(user_turn: str) -> str:
    return user_turn.strip()


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


def build_query(mode: str, history: List[Dict[str, Any]], user_turn: str, max_turns_concat: int) -> str:
    mode = mode.lower().strip()
    if mode == "q1":
        return build_query_q1(user_turn)
    if mode == "q2":
        return build_query_q2_concat(history, user_turn, max_turns_concat=max_turns_concat)
    if mode in {"q3_rule", "q4_t5"}:
        raise NotImplementedError(f"{mode} not implemented yet (we’ll add rewrites in the next files).")
    raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# Formatting (must match training)
# -------------------------
def truncate_text_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def format_model_input(
    tokenizer: Any,
    history: List[Dict[str, Any]],
    user_turn: str,
    evidence_texts: List[str],
    max_passage_tokens: int,
) -> str:
    hist_lines: List[str] = []
    for t in history:
        role = str(t.get("role", "")).strip().lower()
        utt = str(t.get("utterance", "")).strip()
        if not utt:
            continue
        role_label = "User" if role == "user" else "Agent"
        utt = re.sub(r"\s+", " ", utt)
        if len(utt) > 240:
            utt = utt[:237] + "..."
        hist_lines.append(f"{role_label}: {utt}")

    history_block = "\n".join(hist_lines) if hist_lines else "(none)"

    ev_lines: List[str] = []
    for i, ptxt in enumerate(evidence_texts, start=1):
        ptxt = re.sub(r"\s+", " ", ptxt).strip()
        ptxt = truncate_text_to_tokens(tokenizer, ptxt, max_tokens=max_passage_tokens)
        ev_lines.append(f"[{i}] {ptxt}")
    evidence_block = "\n".join(ev_lines) if ev_lines else "(none)"

    return (
        "HISTORY:\n"
        f"{history_block}\n\n"
        "QUESTION:\n"
        f"{user_turn.strip()}\n\n"
        "EVIDENCE:\n"
        f"{evidence_block}\n\n"
        "INSTRUCTION:\n"
        "Write the next agent response. Use the evidence when relevant.\n"
    )


# -------------------------
# Batch generation
# -------------------------
@torch.inference_mode()
def generate_batch(
    model: BartForConditionalGeneration,
    tokenizer: Any,
    prompts: List[str],
    *,
    max_source_len: int,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    enc = tokenizer(
        prompts,
        truncation=True,
        max_length=max_source_len,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to trained model checkpoint dir (e.g., outputs/bart_debug).")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    ap.add_argument("--mode", type=str, choices=["q1", "q2", "q3_rule", "q4_t5"], default="q2")
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--max_turns_concat", type=int, default=6)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_passage_tokens", type=int, default=80)

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=4)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="If >0, limit number of examples for quick debugging.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")

    ap.add_argument("--save_evidence_preview", action="store_true", help="Include a short evidence preview per example (bigger files, easier debugging).")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_dir = Path(args.data_dir).resolve()
    processed = data_dir / "processed"

    examples_path = processed / f"examples_{args.split}.jsonl"
    passages_path = processed / "passages.jsonl"

    if not examples_path.exists():
        raise FileNotFoundError(f"Missing {examples_path}. Did you run step2_build_examples_and_cache.py?")
    if not passages_path.exists():
        raise FileNotFoundError(f"Missing {passages_path}. Did you run step1_build_passages_and_bm25.py?")

    # Output path
    model_name = Path(args.model_dir).name
    out_dir = Path("outputs") / "predictions" / model_name
    safe_mkdir(out_dir)
    out_path = out_dir / f"{args.split}_{args.mode}_k{args.k}.jsonl"

    if out_path.exists() and args.overwrite:
        out_path.unlink()
    if out_path.exists():
        print(f"[WARN] Output exists: {out_path}")
        print("Use --overwrite to regenerate.")
        return

    # Load model/tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading tokenizer/model from:", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = BartForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()
    print("Device:", device)

    # Load passages and build BM25
    print("Loading passages and building BM25...")
    passage_ids, passage_doc_ids, passage_texts = load_passages(passages_path)
    bm25 = BM25Okapi([tokenize_bm25(t) for t in passage_texts])
    pid_to_text = {pid: txt for pid, txt in zip(passage_ids, passage_texts)}
    pid_to_doc = {pid: did for pid, did in zip(passage_ids, passage_doc_ids)}
    print(f"Passages: {len(passage_ids)}")

    # Load examples
    limit = args.limit if args.limit > 0 else None
    examples = load_jsonl(examples_path, limit=limit)
    print(f"Examples loaded: {len(examples)} from {examples_path.name}")

    # Generate
    bs = max(1, args.batch_size)
    buf: List[Dict[str, Any]] = []
    prompts: List[str] = []
    meta: List[Dict[str, Any]] = []

    use_tqdm = tqdm is not None
    it = tqdm(examples, desc=f"Generate {args.split} {args.mode}") if use_tqdm else examples

    for ex in it:  # type: ignore
        ex_id = str(ex["example_id"])
        history = ex["history"]
        user_turn = str(ex["user_turn"])
        reference = str(ex["target"])

        query = build_query(args.mode, history, user_turn, max_turns_concat=args.max_turns_concat)

        # Retrieve top-k passages
        scores = bm25.get_scores(tokenize_bm25(query))
        top = topk_indices_and_scores(scores, args.k)
        top_pids = [passage_ids[i] for i, _ in top]
        evidence_texts = [pid_to_text[pid] for pid in top_pids if pid in pid_to_text]

        prompt = format_model_input(
            tokenizer,
            history=history,
            user_turn=user_turn,
            evidence_texts=evidence_texts,
            max_passage_tokens=args.max_passage_tokens,
        )

        prompts.append(prompt)
        meta.append(
            {
                "example_id": ex_id,
                "query": query,
                "evidence_passage_ids": top_pids,
                "evidence_doc_ids": [pid_to_doc.get(pid, "") for pid in top_pids],
                "user_turn": user_turn,
                "reference": reference,
            }
        )

        if len(prompts) >= bs:
            preds = generate_batch(
                model,
                tokenizer,
                prompts,
                max_source_len=args.max_source_len,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            for m, pred in zip(meta, preds):
                row = dict(m)
                row["prediction"] = pred
                if args.save_evidence_preview:
                    previews = []
                    for pid in row["evidence_passage_ids"][: min(3, len(row["evidence_passage_ids"]))]:
                        txt = pid_to_text.get(pid, "")
                        txt = re.sub(r"\s+", " ", txt).strip()
                        previews.append(txt[:240] + ("..." if len(txt) > 240 else ""))
                    row["evidence_preview"] = previews
                buf.append(row)

            append_jsonl(out_path, buf)
            buf = []
            prompts = []
            meta = []

    # Flush leftovers
    if prompts:
        preds = generate_batch(
            model,
            tokenizer,
            prompts,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        for m, pred in zip(meta, preds):
            row = dict(m)
            row["prediction"] = pred
            if args.save_evidence_preview:
                previews = []
                for pid in row["evidence_passage_ids"][: min(3, len(row["evidence_passage_ids"]))]:
                    txt = pid_to_text.get(pid, "")
                    txt = re.sub(r"\s+", " ", txt).strip()
                    previews.append(txt[:240] + ("..." if len(txt) > 240 else ""))
                row["evidence_preview"] = previews
            buf.append(row)
        append_jsonl(out_path, buf)

    print("\nSaved predictions to:", out_path)
    print("Next file: scoring script for EM / token-F1 / SacreBLEU.")


if __name__ == "__main__":
    main()
