#!/usr/bin/env python3
"""
step5b_eval_retrieval_recall.py

Compute retrieval recall@k using prediction files + gold grounding:
- doc_recall@k: any gold_doc_id appears in evidence_doc_ids
- span_recall@k: any gold_span_id appears in union(span_ids of evidence_passage_ids)

Also reports rewrite applied rate if rewrite_debug exists.

Run:
  python step5b_eval_retrieval_recall.py --data_dir ./data --split val --pred_files ^
    outputs/predictions/bart_debug/val_q1_k5.jsonl ^
    outputs/predictions/bart_debug/val_q2_k5.jsonl ^
    outputs/predictions/bart_debug/val_q3_rule_k5.jsonl ^
    outputs/predictions/bart_debug/val_q4_t5_k5.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    ap.add_argument("--pred_files", nargs="+", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    processed = data_dir / "processed"

    examples_path = processed / f"examples_{args.split}.jsonl"
    lookup_path = processed / "passage_lookup.pkl"

    if not examples_path.exists():
        raise FileNotFoundError(f"Missing {examples_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"Missing {lookup_path}")

    print("Loading examples:", examples_path)
    examples_map = load_examples_map(examples_path)
    print("Examples loaded:", len(examples_map))

    print("Loading passage lookup:", lookup_path)
    with open(lookup_path, "rb") as f:
        passage_lookup: Dict[str, Dict[str, Any]] = pickle.load(f)
    print("Passages in lookup:", len(passage_lookup))

    # Evaluate each predictions file
    for pf in args.pred_files:
        path = Path(pf)
        if not path.exists():
            raise FileNotFoundError(f"Missing pred file: {path}")

        n = 0
        doc_hits = 0
        span_hits = 0

        rewrite_debug_seen = 0
        rewrite_applied = 0

        missing_example = 0

        for row in iter_jsonl(path):
            ex_id = str(row.get("example_id", ""))
            if ex_id not in examples_map:
                missing_example += 1
                continue

            gold_docs: Set[str] = examples_map[ex_id]["gold_doc_ids"]
            gold_spans: Set[str] = examples_map[ex_id]["gold_span_ids"]

            ev_doc_ids = set(str(x) for x in row.get("evidence_doc_ids", []) if x is not None)
            ev_passage_ids = [str(x) for x in row.get("evidence_passage_ids", []) if x is not None]

            # union span ids from evidence passages
            ev_span_ids: Set[str] = set()
            for pid in ev_passage_ids:
                meta = passage_lookup.get(pid)
                if meta and "span_ids" in meta:
                    ev_span_ids |= set(str(s) for s in meta["span_ids"])

            if gold_docs and (gold_docs & ev_doc_ids):
                doc_hits += 1
            if gold_spans and (gold_spans & ev_span_ids):
                span_hits += 1

            # rewrite stats if available
            if "rewrite_debug" in row and isinstance(row["rewrite_debug"], dict):
                rewrite_debug_seen += 1
                if bool(row["rewrite_debug"].get("applied", False)):
                    rewrite_applied += 1

            n += 1

        doc_recall = doc_hits / n if n else 0.0
        span_recall = span_hits / n if n else 0.0
        applied_rate = (rewrite_applied / rewrite_debug_seen) if rewrite_debug_seen else None

        print("\nFILE:", path.name)
        print(f"  n={n} (missing_example_ids={missing_example})")
        print(f"  doc_recall@k={doc_recall:.3f}")
        print(f"  span_recall@k={span_recall:.3f}")
        if applied_rate is not None:
            print(f"  rewrite_applied_rate={applied_rate:.3f} (debug_rows={rewrite_debug_seen})")


if __name__ == "__main__":
    main()
