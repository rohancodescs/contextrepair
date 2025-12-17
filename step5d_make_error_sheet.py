#!/usr/bin/env python3
"""
step5d_make_error_sheet.py

Export a CSV for qualitative error analysis comparing multiple prediction files. The CSV joins:
- examples_{split}.jsonl (history + gold docs/spans)
- passage_lookup.pkl (span ids + passage text previews)
- prediction jsonl files (prediction, query, evidence ids, rewrite_debug)

It can also *sample* and *stratify* by where improved method wins/loses on token-F1. See the CLI parameters in main, and example below:
Example on how to run:
  python step5d_make_error_sheet.py --data_dir ./data --split val ^
    --pred_files outputs/predictions/bart_debug/val_q2_k5.jsonl ^
                outputs/predictions/bart_debug/val_q4_t5_q2fb_append_k5.jsonl ^
    --baseline_idx 0 --improved_idx 1 ^
    --focus trigger --stratify winloss50 --sample 80 ^
    --out_csv outputs/predictions/bart_debug/error_sheet_val_80.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


#IO
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# Normalization + F1
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def f1_score(pred: str, ref: str) -> float:
    pred_toks = normalize_answer(pred).split()
    ref_toks = normalize_answer(ref).split()
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return (2 * precision * recall) / (precision + recall)


#  Trigger slices 
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

PRONOUN_TRIGGERS = {"it", "that", "this", "they", "them", "those", "these", "its", "their"}
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


def in_focus(user_turn: str, focus: str) -> bool:
    trigger, reasons = needs_rewrite(user_turn)
    if focus == "all":
        return True
    if focus == "trigger":
        return trigger
    if focus == "no_trigger":
        return not trigger
    return focus in reasons  # pronoun / short_turn / ellipsis_prefix


# Loaders
def load_examples_map(examples_path: Path) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for ex in iter_jsonl(examples_path):
        ex_id = str(ex["example_id"])
        hist = ex.get("history", [])
        # format history as readable block
        lines = []
        if isinstance(hist, list):
            for t in hist:
                if not isinstance(t, dict):
                    continue
                role = str(t.get("role", "")).lower()
                utt = str(t.get("utterance", "")).strip()
                if not utt:
                    continue
                label = "User" if role == "user" else "Agent"
                lines.append(f"{label}: {utt}")
        history_str = "\n".join(lines)

        m[ex_id] = {
            "history_str": history_str,
            "user_turn": str(ex.get("user_turn", "")).strip(),
            "reference": str(ex.get("target", "")).strip(),
            "gold_doc_ids": set(str(x) for x in ex.get("gold_doc_ids", [])),
            "gold_span_ids": set(str(x) for x in ex.get("gold_span_ids", [])),
        }
    return m


def load_pred_map(pred_path: Path) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for r in iter_jsonl(pred_path):
        ex_id = str(r.get("example_id", ""))
        if not ex_id:
            continue
        m[ex_id] = r
    return m


def preview_passages(passage_lookup: Dict[str, Any], pids: List[str], topn: int = 3, max_chars: int = 220) -> str:
    previews = []
    for pid in pids[:topn]:
        meta = passage_lookup.get(pid, {})
        txt = str(meta.get("text", "")).strip()
        txt = re.sub(r"\s+", " ", txt)
        if len(txt) > max_chars:
            txt = txt[: max_chars - 3] + "..."
        previews.append(txt)
    return " ||| ".join(previews)


def compute_retrieval_hits(
    ex_gold_docs: Set[str],
    ex_gold_spans: Set[str],
    pred_row: Dict[str, Any],
    passage_lookup: Dict[str, Any],
) -> Tuple[bool, bool]:
    ev_doc_ids = set(str(x) for x in pred_row.get("evidence_doc_ids", []) if x is not None)
    ev_pids = [str(x) for x in pred_row.get("evidence_passage_ids", []) if x is not None]

    ev_spans: Set[str] = set()
    for pid in ev_pids:
        meta = passage_lookup.get(pid)
        if meta and "span_ids" in meta:
            ev_spans |= set(str(s) for s in meta["span_ids"])

    doc_hit = bool(ex_gold_docs & ev_doc_ids) if ex_gold_docs else False
    span_hit = bool(ex_gold_spans & ev_spans) if ex_gold_spans else False
    return doc_hit, span_hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    ap.add_argument("--pred_files", nargs="+", required=True)
    ap.add_argument("--baseline_idx", type=int, default=0)
    ap.add_argument("--improved_idx", type=int, default=1)

    ap.add_argument("--focus", type=str, default="trigger",
                    choices=["all", "trigger", "no_trigger", "pronoun", "short_turn", "ellipsis_prefix"])
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--stratify", type=str, default="winloss50", choices=["none", "winloss50"])
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    processed = data_dir / "processed"
    examples_path = processed / f"examples_{args.split}.jsonl"
    lookup_path = processed / "passage_lookup.pkl"

    if not examples_path.exists():
        raise FileNotFoundError(f"Missing {examples_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"Missing {lookup_path}")

    # Load examples + passage lookup
    ex_map = load_examples_map(examples_path)
    with open(lookup_path, "rb") as f:
        passage_lookup: Dict[str, Any] = pickle.load(f)

    # Load prediction maps
    pred_maps = [load_pred_map(Path(p)) for p in args.pred_files]
    names = [Path(p).stem for p in args.pred_files]

    # Intersect example ids across all pred files and examples
    common_ids = set(ex_map.keys())
    for pm in pred_maps:
        common_ids &= set(pm.keys())

    # Apply focus filter
    filtered_ids = []
    for ex_id in common_ids:
        ut = ex_map[ex_id]["user_turn"]
        if in_focus(ut, args.focus):
            filtered_ids.append(ex_id)

    if not filtered_ids:
        raise RuntimeError(f"No examples match focus={args.focus} in the intersection set.")

    # Compute per-example f1 for baseline/improved for stratified sampling
    b_idx = args.baseline_idx
    i_idx = args.improved_idx
    if not (0 <= b_idx < len(pred_maps) and 0 <= i_idx < len(pred_maps)):
        raise ValueError("baseline_idx/improved_idx out of range")

    wins, losses, ties = [], [], []
    for ex_id in filtered_ids:
        ref = ex_map[ex_id]["reference"]
        b_pred = str(pred_maps[b_idx][ex_id].get("prediction", "")).strip()
        i_pred = str(pred_maps[i_idx][ex_id].get("prediction", "")).strip()
        b_f1 = f1_score(b_pred, ref)
        i_f1 = f1_score(i_pred, ref)
        diff = i_f1 - b_f1
        if diff > 1e-12:
            wins.append(ex_id)
        elif diff < -1e-12:
            losses.append(ex_id)
        else:
            ties.append(ex_id)

    # Sample ids
    chosen: List[str] = []
    if args.stratify == "none":
        random.shuffle(filtered_ids)
        chosen = filtered_ids[: min(args.sample, len(filtered_ids))]
    else:
        # win/loss balanced
        half = args.sample // 2
        random.shuffle(wins)
        random.shuffle(losses)
        random.shuffle(ties)
        chosen.extend(wins[: min(half, len(wins))])
        chosen.extend(losses[: min(half, len(losses))])
        # fill remainder from ties then leftover
        remaining = args.sample - len(chosen)
        pool = ties + wins[half:] + losses[half:]
        chosen.extend(pool[: min(remaining, len(pool))])

    # Build CSV rows
    fieldnames = [
        "example_id",
        "focus",
        "trigger_reasons",
        "history",
        "user_turn",
        "reference",
        "gold_doc_ids",
        "gold_span_ids",
    ]

    # Add per-pred-file columns
    for name in names:
        fieldnames.extend([
            f"query__{name}",
            f"prediction__{name}",
            f"doc_hit__{name}",
            f"span_hit__{name}",
            f"evidence_doc_ids__{name}",
            f"evidence_preview__{name}",
            f"rewrite_debug__{name}",
        ])

    # Annotation columns
    fieldnames.extend([
        "label_error_type",
        "label_who_failed",
        "label_notes",
    ])

    out_rows: List[Dict[str, Any]] = []
    for ex_id in chosen:
        ex = ex_map[ex_id]
        ut = ex["user_turn"]
        trigger, reasons = needs_rewrite(ut)

        row: Dict[str, Any] = {
            "example_id": ex_id,
            "focus": args.focus,
            "trigger_reasons": ",".join(reasons),
            "history": ex["history_str"],
            "user_turn": ut,
            "reference": ex["reference"],
            "gold_doc_ids": " | ".join(sorted(ex["gold_doc_ids"])),
            "gold_span_ids": " | ".join(sorted(ex["gold_span_ids"])),
            "label_error_type": "",
            "label_who_failed": "",  # i.e., retrieval, rewrite, generation, reference ambiguity
            "label_notes": "",
        }

        for name, pm in zip(names, pred_maps):
            pr = pm[ex_id]
            query = str(pr.get("query", "")).strip()
            pred = str(pr.get("prediction", "")).strip()
            ev_docs = [str(x) for x in pr.get("evidence_doc_ids", []) if x is not None]
            ev_pids = [str(x) for x in pr.get("evidence_passage_ids", []) if x is not None]

            doc_hit, span_hit = compute_retrieval_hits(ex["gold_doc_ids"], ex["gold_span_ids"], pr, passage_lookup)
            ev_prev = preview_passages(passage_lookup, ev_pids, topn=3, max_chars=220)
            rw_dbg = pr.get("rewrite_debug", "")

            row[f"query__{name}"] = query
            row[f"prediction__{name}"] = pred
            row[f"doc_hit__{name}"] = int(doc_hit)
            row[f"span_hit__{name}"] = int(span_hit)
            row[f"evidence_doc_ids__{name}"] = " | ".join(ev_docs)
            row[f"evidence_preview__{name}"] = ev_prev
            row[f"rewrite_debug__{name}"] = json.dumps(rw_dbg, ensure_ascii=False) if isinstance(rw_dbg, (dict, list)) else str(rw_dbg)

        out_rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved error analysis sheet: {out_path}")
    print(f"Rows: {len(out_rows)} | focus={args.focus} | stratify={args.stratify}")
    print(f"Win/Loss/Tie counts in focus-set: wins={len(wins)} losses={len(losses)} ties={len(ties)}")


if __name__ == "__main__":
    main()
