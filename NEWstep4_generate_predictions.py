#!/usr/bin/env python3
"""
step4_generate_predictions.py (UPDATED)

Generate predictions for a split (val/test/train) under retrieval query modes:
- q1: user turn only
- q2: concat history + user turn (baseline)
- q3_rule: rule-based context repair rewrite -> BM25 query
- q4_t5: neural rewrite (T5) -> BM25 query (optional; you can ignore for now)

Run (debug):
  python step4_generate_predictions.py --data_dir ./data --model_dir ./outputs/bart_debug ^
    --split val --mode q3_rule --k 5 --limit 200 --batch_size 1 --overwrite --save_evidence_preview

Then evaluate:
  python step5_eval_predictions.py --pred_files ^
    outputs/predictions/bart_debug/val_q1_k5.jsonl ^
    outputs/predictions/bart_debug/val_q2_k5.jsonl ^
    outputs/predictions/bart_debug/val_q3_rule_k5.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
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
# Query modes: Q1/Q2
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


# -------------------------
# Rule-based Context Repair (Q3)
# -------------------------
PRONOUN_TRIGGERS = {
    "it", "that", "this", "they", "them", "those", "these", "its", "their", "this", "that",
}


ELLIPSIS_PREFIXES = (
    "what about",
    "how about",
    "and ",
    "besides",
    "besides that",
    "also",
    "in addition",
    "another question",
    "another query",
    "what else",
)


def needs_rewrite(user_turn: str) -> Tuple[bool, List[str]]:
    """
    Conservative trigger detector:
    rewrite only when the turn is likely context-dependent.
    """
    t = user_turn.strip()
    low = t.lower().strip()
    reasons: List[str] = []

    toks = tokenize_bm25(t)
    if len(toks) <= 5:
        reasons.append("short_turn")

    if any(low.startswith(p) for p in ELLIPSIS_PREFIXES):
        reasons.append("ellipsis_prefix")

    # pronoun presence
    if any(tok in PRONOUN_TRIGGERS for tok in toks):
        reasons.append("pronoun")

    # explicit deictic phrases
    if "besides that" in low or "other than that" in low:
        reasons.append("deictic_phrase")

    return (len(reasons) > 0), reasons


@dataclass
class RuleRewriteResult:
    query: str
    applied: bool
    reasons: List[str]
    topic: str


class RuleRewriter:
    """
    Very lightweight, retrieval-oriented rewrite:
    - pick a salient noun phrase from recent history (topic)
    - if turn seems context-dependent, append topic or combine with "what about X" fragments
    """
    def __init__(self) -> None:
        import spacy  # type: ignore

        # We need the parser for noun_chunks; disable NER for speed.
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.stop = self.nlp.Defaults.stop_words

    def _extract_topic(self, history: List[Dict[str, Any]], max_utterances: int = 6) -> str:
        # Look backwards through recent turns for a salient noun phrase.
        # Prefer longer noun chunks; add slight bonus for uppercase acronyms (DMV, SSA).
        recent = history[-max_utterances:] if max_utterances > 0 else history
        best = ("", 0.0)

        # process from most recent to oldest with a recency weight
        recency_weight = 1.0
        for t in reversed(recent):
            utt = str(t.get("utterance", "")).strip()
            if not utt:
                recency_weight *= 0.85
                continue

            doc = self.nlp(utt)
            for chunk in doc.noun_chunks:
                txt = chunk.text.strip()
                txt_low = txt.lower()

                # strip leading articles
                txt = re.sub(r"^(the|a|an)\s+", "", txt, flags=re.I).strip()
                txt_low = txt.lower()

                if not txt:
                    continue
                # filter out pure pronoun / stopword chunks
                toks = [tok for tok in tokenize_bm25(txt) if tok]
                if not toks:
                    continue
                if all((w in self.stop) for w in toks):
                    continue
                if any(w in PRONOUN_TRIGGERS for w in toks) and len(toks) <= 2:
                    continue
                if len(" ".join(toks)) < 3:
                    continue

                bonus = 0.0
                if re.search(r"\b(DMV|SSA|VA)\b", txt, flags=re.I):
                    bonus += 1.5
                if re.search(r"[A-Z]{2,}", txt):
                    bonus += 0.5
                score = recency_weight * (len(toks) + bonus)

                if score > best[1]:
                    best = (txt, score)

            recency_weight *= 0.85

        return best[0]

    def rewrite(self, history: List[Dict[str, Any]], user_turn: str) -> RuleRewriteResult:
        applied, reasons = needs_rewrite(user_turn)
        if not applied:
            return RuleRewriteResult(query=user_turn.strip(), applied=False, reasons=[], topic="")

        topic = self._extract_topic(history)
        if not topic:
            # can't do much without a topic; fall back to concat query
            return RuleRewriteResult(query=user_turn.strip(), applied=False, reasons=reasons + ["no_topic_found"], topic="")

        low = user_turn.lower().strip()

        # Handle "what about X" / "how about X"
        m = re.match(r"^(what about|how about)\s+(.*)$", low)
        if m:
            # Use original casing for tail if possible
            tail = user_turn.strip()[len(m.group(1)):].strip()
            query = f"{topic} {tail}"
            return RuleRewriteResult(query=query.strip(), applied=True, reasons=reasons, topic=topic)

        # Generic: append topic (short and BM25-friendly)
        # Avoid doubling if topic already appears
        if topic.lower() in low:
            query = user_turn.strip()
        else:
            query = f"{user_turn.strip()} {topic}".strip()

        # Keep queries from becoming absurdly long
        if len(query) > 260:
            query = query[:260]

        return RuleRewriteResult(query=query, applied=True, reasons=reasons, topic=topic)


# -------------------------
# (Optional) Neural rewrite (Q4) placeholder
# We'll implement training next; this code is ready once you have a trained rewriter dir.
# -------------------------
class T5Rewriter:
    def __init__(self, rewriter_dir: str, device: str = "cpu") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(rewriter_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(rewriter_dir)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def rewrite(self, history: List[Dict[str, Any]], user_turn: str, max_turns: int = 6) -> str:
        # A simple input format that works well for CANARD-trained T5:
        # "rewrite: <history> question: <q>"
        hist_tail = history[-max_turns:] if max_turns > 0 else history
        hist_str = " <sep> ".join(
            str(t.get("utterance", "")).strip()
            for t in hist_tail
            if str(t.get("utterance", "")).strip()
        )

        if hist_str:
            src = f"rewrite: {hist_str} question: {user_turn.strip()}"
        else:
            src = f"rewrite: question: {user_turn.strip()}"

        enc = self.tok(src, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.generate(
            **enc,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
        )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


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
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    ap.add_argument("--mode", type=str, choices=["q1", "q2", "q3_rule", "q4_t5"], default="q2")
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--max_turns_concat", type=int, default=6)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_passage_tokens", type=int, default=80)

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=4)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--save_evidence_preview", action="store_true")
    ap.add_argument("--save_rewrite_debug", action="store_true", help="Include rewrite debug info in output rows.")
    ap.add_argument(
        "--rewrite_strategy",
        type=str,
        default="append",
        choices=["replace", "append"],
        help="How to use rewrite when triggered: replace baseline query OR append rewrite to baseline.",
    )
    ap.add_argument(
        "--rewrite_fallback",
        type=str,
        default="q2",
        choices=["q1", "q2"],
        help="What query to use when rewrite isn't applied (or fails). Default should be q2.",
    )
    ap.add_argument("--run_tag", type=str, default="", help="Optional tag added to output filename.")

    # q4 args (optional for now)
    ap.add_argument("--t5_rewriter_dir", type=str, default="", help="Path to trained T5 rewriter dir for q4_t5.")
    ap.add_argument("--rewriter_device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--rewrite_only_if_trigger", action="store_true", help="Only apply rewrite if trigger fires (recommended).")

    args = ap.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_dir = Path(args.data_dir).resolve()
    processed = data_dir / "processed"

    examples_path = processed / f"examples_{args.split}.jsonl"
    passages_path = processed / "passages.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"Missing {examples_path}.")
    if not passages_path.exists():
        raise FileNotFoundError(f"Missing {passages_path}.")

    # Output
    model_name = Path(args.model_dir).name
    out_dir = Path("outputs") / "predictions" / model_name
    safe_mkdir(out_dir)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out_path = out_dir / f"{args.split}_{args.mode}{tag}_k{args.k}.jsonl"

    if out_path.exists() and args.overwrite:
        out_path.unlink()
    if out_path.exists():
        print(f"[WARN] Output exists: {out_path} (use --overwrite).")
        return

    # Load generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading generator from:", args.model_dir)
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    gen_model = BartForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    gen_model.eval()
    print("Generator device:", device)

    # Load passages + BM25
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

    # Rewriters (only initialize if needed)
    rule_rewriter: Optional[RuleRewriter] = None
    t5_rewriter: Optional[T5Rewriter] = None

    if args.mode == "q3_rule":
        rule_rewriter = RuleRewriter()

    if args.mode == "q4_t5":
        if not args.t5_rewriter_dir:
            raise ValueError("q4_t5 requires --t5_rewriter_dir pointing to a trained T5 rewriter.")
        t5_rewriter = T5Rewriter(args.t5_rewriter_dir, device=args.rewriter_device)

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

        # Compute both Q1 and Q2 once, then determine baseline
        q1_base = build_query_q1(user_turn)
        q2_base = build_query_q2_concat(history, user_turn, max_turns_concat=args.max_turns_concat)
        baseline = q2_base if args.rewrite_fallback == "q2" else q1_base

        # Build retrieval query depending on mode
        rewrite_debug: Dict[str, Any] = {}
        if args.mode == "q1":
            query = build_query_q1(user_turn)

        elif args.mode == "q2":
            query = build_query_q2_concat(history, user_turn, max_turns_concat=args.max_turns_concat)

        elif args.mode == "q3_rule":
            assert rule_rewriter is not None
            do_rewrite, reasons = needs_rewrite(user_turn)

            if args.rewrite_only_if_trigger and not do_rewrite:
                query = baseline
                rewrite_debug = {"applied": False, "reasons": [], "topic": "", "strategy": args.rewrite_strategy, "fallback": args.rewrite_fallback}
            else:
                rr = rule_rewriter.rewrite(history, user_turn)
                rewrite_str = rr.query if rr.applied else ""

                if not rewrite_str:
                    query = baseline
                else:
                    query = rewrite_str if args.rewrite_strategy == "replace" else (baseline + " " + rewrite_str)

                rewrite_debug = {"applied": rr.applied, "reasons": rr.reasons, "topic": rr.topic, "strategy": args.rewrite_strategy, "fallback": args.rewrite_fallback}

        elif args.mode == "q4_t5":
            assert t5_rewriter is not None
            do_rewrite, reasons = needs_rewrite(user_turn)

            if args.rewrite_only_if_trigger and not do_rewrite:
                query = baseline
                rewrite_debug = {"applied": False, "reasons": [], "rewrite": "", "strategy": args.rewrite_strategy, "fallback": args.rewrite_fallback}
            else:
                rewrite = t5_rewriter.rewrite(history, user_turn, max_turns=args.max_turns_concat).strip()

                if not rewrite:
                    query = baseline
                    applied = False
                else:
                    query = rewrite if args.rewrite_strategy == "replace" else (baseline + " " + rewrite)
                    applied = True

                rewrite_debug = {"applied": applied, "reasons": reasons, "rewrite": rewrite, "strategy": args.rewrite_strategy, "fallback": args.rewrite_fallback}

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Retrieve top-k
        scores = bm25.get_scores(tokenize_bm25(query))
        top = topk_indices_and_scores(scores, args.k)
        top_pids = [passage_ids[i] for i, _ in top]
        evidence_texts = [pid_to_text[pid] for pid in top_pids if pid in pid_to_text]

        prompt = format_model_input(
            gen_tokenizer,
            history=history,
            user_turn=user_turn,
            evidence_texts=evidence_texts,
            max_passage_tokens=args.max_passage_tokens,
        )

        prompts.append(prompt)
        mrow: Dict[str, Any] = {
            "example_id": ex_id,
            "mode": args.mode,
            "query": query,
            "user_turn": user_turn,
            "reference": reference,
            "evidence_passage_ids": top_pids,
            "evidence_doc_ids": [pid_to_doc.get(pid, "") for pid in top_pids],
        }
        if args.save_rewrite_debug and args.mode in {"q3_rule", "q4_t5"}:
            mrow["rewrite_debug"] = rewrite_debug

        meta.append(mrow)

        if len(prompts) >= bs:
            preds = generate_batch(
                gen_model,
                gen_tokenizer,
                prompts,
                max_source_len=args.max_source_len,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            out_rows: List[Dict[str, Any]] = []
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
                out_rows.append(row)

            append_jsonl(out_path, out_rows)
            prompts, meta = [], []

    # flush
    if prompts:
        preds = generate_batch(
            gen_model,
            gen_tokenizer,
            prompts,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        out_rows = []
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
            out_rows.append(row)
        append_jsonl(out_path, out_rows)

    print("\nSaved predictions to:", out_path)
    print("Next: score with step5_eval_predictions.py (add this file to the list).")


if __name__ == "__main__":
    main()
