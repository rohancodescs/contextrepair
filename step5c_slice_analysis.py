#!/usr/bin/env python3
"""
step5c_slice_analysis.py

Slice evaluation for generation + retrieval metrics.
Slices are based on the same heuristic triggers used by the rewriter:
- short turns
- ellipsis prefixes ("what about", "besides that", etc.)
- pronouns/demonstratives ("it", "that", "they", ...)

For each predictions JSONL file:
- compute EM/F1/SacreBLEU overall
- compute doc_recall@k and span_recall@k overall
- compute same metrics on slices (trigger / no_trigger / pronoun / ellipsis / short)

Run:
  python step5c_slice_analysis.py --data_dir ./data --split val --pred_files ^
    outputs/predictions/bart_debug/val_q2_k5.jsonl ^
    outputs/predictions/bart_debug/val_q4_t5_q2fb_append_k5.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None


# -------------------------
# IO
# -------------------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_examples_map(examples_path: Path) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for ex in iter_jsonl(examples_path):
        ex_id = str(ex["example_id"])
        m[ex_id] = {
            "gold_doc_ids": set(str(x) for x in ex.get("gold_doc_ids", [])),
            "gold_span_ids": set(str(x) for x in ex.get("gold_span_ids", [])),
        }
    return m


# -------------------------
# Tokenization + triggers (match your step4 logic)
# -------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

PRONOUN_TRIGGERS = {
    "it", "that", "this", "they", "them", "those", "these", "its", "their",
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


def tokenize_bm25(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def needs_rewrite(user_turn: str) -> Tuple[bool, List[str]]:
    t = user_turn.strip()
    low = t.lower().strip()
    reasons: List[str] = []

    toks = tokenize_bm25(t)
    if len(toks) <= 5:
        reasons.append("short_turn")
    if any(low.startswith(p) for p in ELLIPSIS_PREFIXES):
        reasons.append("ellipsis_prefix")
    if any(tok in PRONOUN_TRIGGERS for tok in toks):
        reasons.append("pronoun")
    if "besides that" in low or "other than that" in low:
        reasons.append("deictic_phrase")

    return (len(reasons) > 0), reasons


# -------------------------
# EM/F1 normalization (same as step5_eval_predictions)
# -------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(ref) else 0.0


def f1_score(pred: str, ref: str) -> float:
    pred_toks = normalize_answer(pred).split()
    ref_toks = normalize_answer(ref).split()

    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0

    from collections import Counter
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return (2 * precision * recall) / (precision + recall)


# -------------------------
# Slice accumulator
# -------------------------
class SliceAgg:
    def __init__(self) -> None:
        self.n = 0
        self.em_sum = 0.0
        self.f1_sum = 0.0
        self.preds: List[str] = []
        self.refs: List[str] = []
        self.doc_hits = 0
        self.span_hits = 0

    def add(
        self,
        pred: str,
        ref: str,
        doc_hit: bool,
        span_hit: bool,
    ) -> None:
        self.n += 1
        self.em_sum += exact_match(pred, ref)
        self.f1_sum += f1_score(pred, ref)
        self.preds.append(pred)
        self.refs.append(ref)
        self.doc_hits += int(doc_hit)
        self.span_hits += int(span_hit)

    def finalize(self) -> Dict[str, Any]:
        if self.n == 0:
            return {
                "n": 0,
                "em": 0.0,
                "f1": 0.0,
                "sacrebleu": None,
                "doc_recall": 0.0,
                "span_recall": 0.0,
            }

        bleu = None
        if sacrebleu is not None:
            bleu = float(sacrebleu.corpus_bleu(self.preds, [self.refs]).score)

        return {
            "n": self.n,
            "em": self.em_sum / self.n,
            "f1": self.f1_sum / self.n,
            "sacrebleu": bleu,
            "doc_recall": self.doc_hits / self.n,
            "span_recall": self.span_hits / self.n,
        }


def print_slice_table(file_name: str, out: Dict[str, Dict[str, Any]]) -> None:
    # Print a compact table for key slices
    slices_order = ["all", "trigger", "no_trigger", "pronoun", "ellipsis_prefix", "short_turn"]
    print("\nFILE:", file_name)
    header = ["slice", "n", "EM", "F1", "BLEU", "docR@k", "spanR@k"]
    print(" | ".join(h.ljust(12) for h in header))
    print("-" * (12 * len(header) + 3 * (len(header) - 1)))

    for s in slices_order:
        r = out.get(s, None)
        if r is None:
            continue
        bleu = r["sacrebleu"]
        bleu_str = "N/A" if bleu is None else f"{bleu:.2f}"
        print(
            f"{s.ljust(12)} | "
            f"{str(r['n']).ljust(12)} | "
            f"{r['em']:.4f}".ljust(12) + " | "
            f"{r['f1']:.4f}".ljust(12) + " | "
            f"{bleu_str}".ljust(12) + " | "
            f"{r['doc_recall']:.3f}".ljust(12) + " | "
            f"{r['span_recall']:.3f}".ljust(12)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    ap.add_argument("--pred_files", nargs="+", required=True)
    ap.add_argument("--save_json", type=str, default="", help="Optional: save slice metrics to JSON.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    processed = data_dir / "processed"
    examples_path = processed / f"examples_{args.split}.jsonl"
    lookup_path = processed / "passage_lookup.pkl"

    if not examples_path.exists():
        raise FileNotFoundError(f"Missing {examples_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"Missing {lookup_path}")

    if sacrebleu is None:
        print("[WARN] sacrebleu not installed, BLEU will be N/A (pip install sacrebleu).")

    examples_map = load_examples_map(examples_path)

    with open(lookup_path, "rb") as f:
        passage_lookup: Dict[str, Dict[str, Any]] = pickle.load(f)

    all_results: Dict[str, Any] = {}

    for pf in args.pred_files:
        path = Path(pf)
        if not path.exists():
            raise FileNotFoundError(f"Missing pred file: {path}")

        # slice -> aggregator
        aggs: Dict[str, SliceAgg] = defaultdict(SliceAgg)

        for row in iter_jsonl(path):
            ex_id = str(row.get("example_id", ""))
            if ex_id not in examples_map:
                continue

            pred = str(row.get("prediction", "")).strip()
            ref = str(row.get("reference", "")).strip()
            user_turn = str(row.get("user_turn", "")).strip()

            gold_docs: Set[str] = examples_map[ex_id]["gold_doc_ids"]
            gold_spans: Set[str] = examples_map[ex_id]["gold_span_ids"]

            ev_doc_ids = set(str(x) for x in row.get("evidence_doc_ids", []) if x is not None)
            ev_passage_ids = [str(x) for x in row.get("evidence_passage_ids", []) if x is not None]

            ev_span_ids: Set[str] = set()
            for pid in ev_passage_ids:
                meta = passage_lookup.get(pid)
                if meta and "span_ids" in meta:
                    ev_span_ids |= set(str(s) for s in meta["span_ids"])

            doc_hit = bool(gold_docs & ev_doc_ids) if gold_docs else False
            span_hit = bool(gold_spans & ev_span_ids) if gold_spans else False

            trigger, reasons = needs_rewrite(user_turn)

            # Always add to overall
            aggs["all"].add(pred, ref, doc_hit, span_hit)

            if trigger:
                aggs["trigger"].add(pred, ref, doc_hit, span_hit)
            else:
                aggs["no_trigger"].add(pred, ref, doc_hit, span_hit)

            # reason slices
            if "pronoun" in reasons:
                aggs["pronoun"].add(pred, ref, doc_hit, span_hit)
            if "ellipsis_prefix" in reasons:
                aggs["ellipsis_prefix"].add(pred, ref, doc_hit, span_hit)
            if "short_turn" in reasons:
                aggs["short_turn"].add(pred, ref, doc_hit, span_hit)

        finalized = {k: v.finalize() for k, v in aggs.items()}
        all_results[path.name] = finalized
        print_slice_table(path.name, finalized)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print("\nSaved slice metrics JSON to:", out_path)


if __name__ == "__main__":
    main()
