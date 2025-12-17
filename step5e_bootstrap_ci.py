#!/usr/bin/env python3
"""
step5e_bootstrap_ci.py

Bootstrap confidence intervals for delta metrics between two prediction files, it calculates the per-example EM and token-F1

Run:
  python step5e_bootstrap_ci.py --pred_a outputs/predictions/bart_full/test_q2_k5.jsonl ^
    --pred_b outputs/predictions/bart_full/test_q4_t5_q2fb_append_k5.jsonl ^
    --n_boot 2000 --seed 13
"""

from __future__ import annotations
import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

#  IO 
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_map(path: Path) -> Dict[str, Dict[str, Any]]:
    m = {}
    for r in iter_jsonl(path):
        ex_id = str(r.get("example_id", ""))
        if ex_id:
            m[ex_id] = r
    return m

# EM and F1 metrics
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
    pt = normalize_answer(pred).split()
    rt = normalize_answer(ref).split()
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    from collections import Counter
    common = Counter(pt) & Counter(rt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(rt)
    return (2 * precision * recall) / (precision + recall)

def percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    if hi == lo:
        return xs[lo]
    w = k - lo
    return xs[lo] * (1 - w) + xs[hi] * w

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_a", type=str, required=True, help="Baseline file (e.g., q2)")
    ap.add_argument("--pred_b", type=str, required=True, help="Improved file (e.g., q4)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)

    a = load_map(Path(args.pred_a))
    b = load_map(Path(args.pred_b))
    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        raise RuntimeError("No overlapping example_ids between files.")

    em_a, em_b, f1_a, f1_b = [], [], [], []
    for ex_id in common:
        ra = a[ex_id]
        rb = b[ex_id]
        ref = str(ra.get("reference", rb.get("reference", ""))).strip()
        pa = str(ra.get("prediction", "")).strip()
        pb = str(rb.get("prediction", "")).strip()
        em_a.append(exact_match(pa, ref))
        em_b.append(exact_match(pb, ref))
        f1_a.append(f1_score(pa, ref))
        f1_b.append(f1_score(pb, ref))

    def mean(xs): return sum(xs)/len(xs)

    delta_em = mean(em_b) - mean(em_a)
    delta_f1 = mean(f1_b) - mean(f1_a)

    # bootstrap deltas
    deltas_em, deltas_f1 = [], []
    n = len(common)
    for _ in range(args.n_boot):
        idxs = [random.randrange(n) for _ in range(n)]
        deltas_em.append(mean([em_b[i] for i in idxs]) - mean([em_a[i] for i in idxs]))
        deltas_f1.append(mean([f1_b[i] for i in idxs]) - mean([f1_a[i] for i in idxs]))

    ci_em = (percentile(deltas_em, 0.025), percentile(deltas_em, 0.975))
    ci_f1 = (percentile(deltas_f1, 0.025), percentile(deltas_f1, 0.975))

    # two-sided bootstrap “p-ish” value: fraction of deltas crossing 0
    p_em = 2 * min(sum(d <= 0 for d in deltas_em)/len(deltas_em), sum(d >= 0 for d in deltas_em)/len(deltas_em))
    p_f1 = 2 * min(sum(d <= 0 for d in deltas_f1)/len(deltas_f1), sum(d >= 0 for d in deltas_f1)/len(deltas_f1))

    print(f"n={n}")
    print(f"ΔEM = {delta_em:.4f}  95%CI[{ci_em[0]:.4f}, {ci_em[1]:.4f}]  approx_p={p_em:.4f}")
    print(f"ΔF1 = {delta_f1:.4f}  95%CI[{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]  approx_p={p_f1:.4f}")

if __name__ == "__main__":
    main()
