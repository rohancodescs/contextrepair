#!/usr/bin/env python3
"""
step5h_extract_case_studies.py

Extract top-N win/loss examples between two prediction files (baseline vs improved).
Outputs a readable text report you can paste into your paper/slides.

Run:
  python step5h_extract_case_studies.py --pred_a outputs/predictions/bart_debug/val_q2_lim1000_k5.jsonl ^
    --pred_b outputs/predictions/bart_debug/val_q4_t5_lim1000_q2fb_append_k5.jsonl ^
    --out_txt outputs/predictions/bart_debug/case_studies_val_lim1000.txt ^
    --top_n 6 --focus trigger
"""

from __future__ import annotations
import argparse, json, re, string
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

# -------- IO --------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def load_map(path: Path) -> Dict[str, Dict[str, Any]]:
    m={}
    for r in iter_jsonl(path):
        ex_id=str(r.get("example_id",""))
        if ex_id:
            m[ex_id]=r
    return m

# -------- F1 --------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    s=(s or "").lower()
    s=s.translate(_PUNCT_TABLE)
    s=_ARTICLES.sub(" ", s)
    s=_WS.sub(" ", s).strip()
    return s

def f1_score(pred: str, ref: str) -> float:
    pt = normalize_answer(pred).split()
    rt = normalize_answer(ref).split()
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    common = Counter(pt) & Counter(rt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(rt)
    return (2 * precision * recall) / (precision + recall)

# -------- Trigger detector (matches your project) --------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PRONOUN_TRIGGERS = {"it","that","this","they","them","those","these","its","their"}
ELLIPSIS_PREFIXES = (
    "what about","how about","and ","besides","besides that","also",
    "in addition","another question","another query","what else",
)

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())

def needs_rewrite(user_turn: str) -> bool:
    low = (user_turn or "").strip().lower()
    toks = tokenize(user_turn)
    reasons = 0
    if len(toks) <= 5:
        reasons += 1
    if any(low.startswith(p) for p in ELLIPSIS_PREFIXES):
        reasons += 1
    if any(t in PRONOUN_TRIGGERS for t in toks):
        reasons += 1
    if "besides that" in low or "other than that" in low:
        reasons += 1
    return reasons > 0

def fmt_preview(r: Dict[str, Any]) -> str:
    prev = r.get("evidence_preview", [])
    if isinstance(prev, list):
        return "\n".join([f"  - {p}" for p in prev])
    return str(prev)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_a", required=True)
    ap.add_argument("--pred_b", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--top_n", type=int, default=6)
    ap.add_argument("--focus", choices=["all","trigger","no_trigger"], default="all")
    args = ap.parse_args()

    a = load_map(Path(args.pred_a))
    b = load_map(Path(args.pred_b))
    common = sorted(set(a.keys()) & set(b.keys()))
    rows: List[Tuple[float,str]] = []

    for ex_id in common:
        ra, rb = a[ex_id], b[ex_id]
        ut = str(ra.get("user_turn",""))
        if args.focus == "trigger" and not needs_rewrite(ut):
            continue
        if args.focus == "no_trigger" and needs_rewrite(ut):
            continue

        ref = str(ra.get("reference", rb.get("reference","")))
        pa = str(ra.get("prediction",""))
        pb = str(rb.get("prediction",""))
        da = f1_score(pa, ref)
        db = f1_score(pb, ref)
        rows.append((db-da, ex_id))

    rows.sort(reverse=True)
    top_wins = rows[:args.top_n]
    top_losses = sorted(rows, key=lambda x: x[0])[:args.top_n]

    out = []
    out.append(f"Case studies focus={args.focus} | total={len(rows)}")
    out.append("\n=== TOP WINS (B - A) ===")
    for d, ex_id in top_wins:
        ra, rb = a[ex_id], b[ex_id]
        out.append(f"\nΔF1={d:.4f}  example_id={ex_id}")
        out.append(f"USER: {ra.get('user_turn','')}")
        out.append(f"REF:  {ra.get('reference','')}")
        out.append(f"A(pred): {ra.get('prediction','')}")
        out.append(f"B(pred): {rb.get('prediction','')}")
        out.append(f"A(query): {ra.get('query','')}")
        out.append(f"B(query): {rb.get('query','')}")
        out.append("B(rewrite_debug): " + json.dumps(rb.get("rewrite_debug", {}), ensure_ascii=False))
        out.append("B(evidence_preview):\n" + fmt_preview(rb))

    out.append("\n=== TOP LOSSES (B - A) ===")
    for d, ex_id in top_losses:
        ra, rb = a[ex_id], b[ex_id]
        out.append(f"\nΔF1={d:.4f}  example_id={ex_id}")
        out.append(f"USER: {ra.get('user_turn','')}")
        out.append(f"REF:  {ra.get('reference','')}")
        out.append(f"A(pred): {ra.get('prediction','')}")
        out.append(f"B(pred): {rb.get('prediction','')}")
        out.append(f"A(query): {ra.get('query','')}")
        out.append(f"B(query): {rb.get('query','')}")
        out.append("B(rewrite_debug): " + json.dumps(rb.get("rewrite_debug", {}), ensure_ascii=False))
        out.append("B(evidence_preview):\n" + fmt_preview(rb))

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(out), encoding="utf-8")
    print("Wrote:", args.out_txt)

if __name__ == "__main__":
    main()
