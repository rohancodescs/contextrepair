#!/usr/bin/env python3
"""
step5_eval_predictions.py

Evaluate prediction JSONL files produced by step4_generate_predictions.py.

Metrics generated include:
- Exact Match (EM) using normalized strings (lowercase, strip punctuation/articles)
- Token-level F1 (SQuAD-style) using normalized tokens
- SacreBLEU (corpus_bleu) using raw strings (strip only)

Run:
  python step5_eval_predictions.py --pred_files outputs/predictions/bart_debug/val_q1_k5.jsonl `
    outputs/predictions/bart_debug/val_q2_k5.jsonl --save_json outputs/predictions/bart_debug/metrics_val_200.json
"""

from __future__ import annotations
import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import sacrebleu # type: ignore
except Exception as e:
    sacrebleu = None


# IO Streams prediction rows from a JSONL file.
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)



# Normalization for EM/F1
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")

# Applies lowercase/article/punctuation normalization to answers.
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

# Computes SQuAD-style token F1 between prediction and reference.
def f1_score(prediction: str, reference: str) -> float:
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    pred_tokens = pred_norm.split() if pred_norm else []
    ref_tokens = ref_norm.split() if ref_norm else []

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # count overlap
    from collections import Counter
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


# Returns 1.0 if normalized prediction matches reference exactly.
def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0

# Computes EM/F1/BLEU metrics for a prediction JSONL file.
def evaluate_file(path: Path) -> Dict[str, Any]:
    rows = list(iter_jsonl(path))
    if not rows:
        return {"file": str(path), "n": 0, "em": 0.0, "f1": 0.0, "sacrebleu": None}

    preds: List[str] = []
    refs: List[str] = []

    ems: List[float] = []
    f1s: List[float] = []

    missing = 0
    for r in rows:
        pred = r.get("prediction", None)
        ref = r.get("reference", None)
        if pred is None or ref is None:
            missing += 1
            continue
        pred_s = str(pred).strip()
        ref_s = str(ref).strip()
        preds.append(pred_s)
        refs.append(ref_s)
        ems.append(exact_match(pred_s, ref_s))
        f1s.append(f1_score(pred_s, ref_s))

    n = len(preds)
    em = sum(ems) / n if n else 0.0
    f1 = sum(f1s) / n if n else 0.0

    bleu = None
    if sacrebleu is not None and n > 0:
        # SacreBLEU expects list of system outputs and list-of-reference-lists
        bleu_obj = sacrebleu.corpus_bleu(preds, [refs])
        bleu = float(bleu_obj.score)

    return {
        "file": str(path),
        "n": n,
        "skipped_missing_pred_or_ref": missing,
        "em": em,
        "f1": f1,
        "sacrebleu": bleu,
    }


# Prints a simple metrics table for multiple runs.
def print_table(results: List[Dict[str, Any]]) -> None:
    # Simple console table
    headers = ["file", "n", "EM", "F1", "SacreBLEU"]
    rows = []
    for r in results:
        fname = Path(r["file"]).name
        rows.append([
            fname,
            str(r.get("n", 0)),
            f"{r.get('em', 0.0):.4f}",
            f"{r.get('f1', 0.0):.4f}",
            "N/A" if r.get("sacrebleu", None) is None else f"{r['sacrebleu']:.2f}",
        ])

    col_widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    # Formats a table row with padded columns.
    def fmt_row(items: List[str]) -> str:
        return " | ".join(items[i].ljust(col_widths[i]) for i in range(len(items)))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))
    print()


# parses CLI args and evaluates each requested predictions file
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_files", nargs="+", required=True, help="One or more prediction JSONL files.")
    ap.add_argument("--save_json", type=str, default="", help="Optional path to save metrics JSON.")
    args = ap.parse_args()

    pred_paths = [Path(p) for p in args.pred_files]
    for p in pred_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing prediction file: {p}")

    if sacrebleu is None:
        print(" sacrebleu import failed. SacreBLEU will be N/A.")
        print("  Install with: pip install sacrebleu")

    results = [evaluate_file(p) for p in pred_paths]
    print_table(results)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        print("Saved metrics JSON to:", out_path)


if __name__ == "__main__":
    main()
