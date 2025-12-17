"""
step1_build_passages_and_bm25.py

Builds the passage corpus (structure-aware sections from MultiDoc2Dial) + BM25 index, then run a quick retrieval for evaluation.

To run the file: python step1_build_passages_and_bm25.py --data_dir ./data --k 5

Outputs (in outputs folder):
  data/processed/passages.jsonl
  data/processed/passage_meta.json
  data/indices/bm25_index.pkl

Notes:
- We build passages by grouping spans by (doc_id, id_sec) and using text_se, which makes span-level recall@k easy later
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi  

#regex
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# Creates the directory path if it does not already exist.
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Loads JSON data from disk.
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Writes a Python object to disk as JSON.
def write_json(path: Path, obj: Any) -> None:
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# Streams an iterable of dictionaries to a JSONL file.
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Performs a simple lowercase alphanumeric tokenization.
def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


# Recursively finds the first JSON file whose name matches the specified pattern.
def find_first_file(root: Path, patterns: List[str]) -> Optional[Path]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for p in root.rglob("*.json"):
        name = p.name
        if any(rx.search(name) for rx in compiled):
            return p
    return None


# Locates the docs/train/val JSON files produced by step0.
def locate_multidoc2dial_files(data_dir: Path) -> Tuple[Path, Path, Path]:
    # find docs/train/val json files in ~/raw/
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Expected {raw_dir} to exist. Did you run step0_bootstrap.py?")

    docs_json = find_first_file(
        raw_dir,
        patterns=[r"(multi|mutl)doc2dial.*_doc\.json$"],
    )
    dial_train_json = find_first_file(
        raw_dir,
        patterns=[r"(multi|mutl)doc2dial.*dial.*train.*\.json$"],
    )
    dial_val_json = find_first_file(
        raw_dir,
        patterns=[r"(multi|mutl)doc2dial.*dial.*valid.*\.json$", r"(multi|mutl)doc2dial.*dial.*validation.*\.json$"],
    )

    if not docs_json or not dial_train_json or not dial_val_json:
        raise FileNotFoundError(
            f"Could not locate required files under {raw_dir}.\n"
            f"docs_json={docs_json}\ntrain={dial_train_json}\nval={dial_val_json}"
        )
    return docs_json, dial_train_json, dial_val_json


# pulls nested dict contents if the target key exists.
def unwrap_if_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return obj


# Loads MultiDoc2Dial documents and indexes them by domain/doc_id.
def load_docs(docs_json: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    raw = load_json(docs_json)
    raw = unwrap_if_key(raw, "doc_data")
    if not isinstance(raw, dict):
        raise ValueError(f"docs_json root must be dict after unwrap; got {type(raw)}")
      
    # expected outcome: domain -> doc_id -> doc_obj
    docs_by_domain: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for domain, v in raw.items():
        if not isinstance(v, dict):
            if isinstance(v, list):
                tmp: Dict[str, Dict[str, Any]] = {}
                for doc in v:
                    if isinstance(doc, dict) and "doc_id" in doc:
                        tmp[str(doc["doc_id"])] = doc
                docs_by_domain[str(domain)] = tmp
            else:
                continue
        else:
            docs_by_domain[str(domain)] = {str(doc_id): doc for doc_id, doc in v.items() if isinstance(doc, dict)}

    total_docs = sum(len(x) for x in docs_by_domain.values())
    if total_docs == 0:
        raise ValueError("Parsed 0 documents. Check docs_json format.")
    return docs_by_domain


# Loads dialogue JSON and returns a consistent domain to dialogues mapping
def load_dialogues(dial_json: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw = load_json(dial_json)
    raw = unwrap_if_key(raw, "dial_data")
    if not isinstance(raw, dict):
        raise ValueError(f"dial_json root must be dict after unwrap; got {type(raw)}")

    out: Dict[str, List[Dict[str, Any]]] = {}
    for domain, v in raw.items():
        if isinstance(v, list):
            out[str(domain)] = [d for d in v if isinstance(d, dict)]
        elif isinstance(v, dict):
            out[str(domain)] = [d for d in v.values() if isinstance(d, dict)]
        else:
            out[str(domain)] = []
    return out


# returns agent-referenced examples with history and gold doc/span ids
def iter_agent_examples(
    dialogues_by_domain: Dict[str, List[Dict[str, Any]]],
    history_turns: int = 6,
) -> Iterable[Tuple[str, List[Dict[str, Any]], str, str, Set[str], Set[str]]]:

    for domain, dials in dialogues_by_domain.items():
        for dial in dials:
            dial_id = str(dial.get("dial_id", ""))
            turns = dial.get("turns", [])
            if not isinstance(turns, list):
                continue

            for i, t in enumerate(turns):
                if not isinstance(t, dict):
                    continue
                if str(t.get("role", "")).lower() != "agent":
                    continue
                refs = t.get("references", [])
                if not refs:
                    continue

                # find nearest preceding user turn
                j = i - 1
                while j >= 0:
                    tj = turns[j]
                    if isinstance(tj, dict) and str(tj.get("role", "")).lower() == "user":
                        break
                    j -= 1
                if j < 0:
                    continue

                user_turn = str(turns[j].get("utterance", "")).strip()
                agent_turn = str(t.get("utterance", "")).strip()
                if not user_turn or not agent_turn:
                    continue

                # history excludes current user turn
                # include up to history_turns previous turns
                hist_start = max(0, j - history_turns)
                hist = [x for x in turns[hist_start:j] if isinstance(x, dict)]

                gold_doc_ids: Set[str] = set()
                gold_span_ids: Set[str] = set()
                for r in refs:
                    if not isinstance(r, dict):
                        continue
                    if "doc_id" in r and r["doc_id"] is not None:
                        gold_doc_ids.add(str(r["doc_id"]))
                    if "id_sp" in r and r["id_sp"] is not None:
                        gold_span_ids.add(str(r["id_sp"]))

                if not gold_doc_ids and not gold_span_ids:
                    continue

                ex_id = f"{domain}::{dial_id}::turn{i}"
                yield ex_id, hist, user_turn, agent_turn, gold_doc_ids, gold_span_ids

@dataclass
class Passage:
    passage_id: str
    domain: str
    doc_id: str
    id_sec: str
    title_path: List[str]
    text: str
    span_ids: List[str]


def build_section_passages(
    docs_by_domain: Dict[str, Dict[str, Dict[str, Any]]],
    min_chars: int = 60,
) -> List[Passage]:

    passages: List[Passage] = []
    pid = 0

    for domain, docs in docs_by_domain.items():
        for doc_id, doc in docs.items():
            spans = doc.get("spans", {})
            if not isinstance(spans, dict):
                continue
              
            sec_map: Dict[str, Dict[str, Any]] = {}

            for id_sp, sp in spans.items():
                if not isinstance(sp, dict):
                    continue
                id_sec = sp.get("id_sec")
                text_sec = sp.get("text_sec")
                title = sp.get("title")
                parent_titles = sp.get("parent_titles", [])

                if id_sec is None or text_sec is None:
                    continue

                id_sec_s = str(id_sec)
                entry = sec_map.get(id_sec_s)
                if entry is None:
                    entry = {
                        "text_sec": str(text_sec),
                        "title": str(title) if title is not None else "",
                        "parent_titles": [str(x) for x in parent_titles] if isinstance(parent_titles, list) else [],
                        "span_ids": [],
                    }
                    sec_map[id_sec_s] = entry

                entry["span_ids"].append(str(id_sp))

            # create passages
            doc_title = str(doc.get("title", "")).strip()
            for id_sec_s, entry in sec_map.items():
                text = str(entry.get("text_sec", "")).strip()
                text = re.sub(r"\s+", " ", text)
                if len(text) < min_chars:
                    continue

                title_path = []
                if isinstance(entry.get("parent_titles"), list):
                    title_path.extend([t for t in entry["parent_titles"] if t])
                if entry.get("title"):
                    title_path.append(str(entry["title"]))

              
                # this constant is needed across experiments
                prefix_parts = []
                if doc_title:
                    prefix_parts.append(doc_title)
                if title_path:
                    prefix_parts.append(" > ".join(title_path))
                prefix = " | ".join(prefix_parts)
                full_text = f"{prefix} :: {text}" if prefix else text

                passages.append(
                    Passage(
                        passage_id=f"p{pid}",
                        domain=domain,
                        doc_id=str(doc_id),
                        id_sec=id_sec_s,
                        title_path=title_path,
                        text=full_text,
                        span_ids=[str(x) for x in entry.get("span_ids", [])],
                    )
                )
                pid += 1

    if not passages:
        raise ValueError("Built 0 passages. Check docs/spans parsing.")
    return passages


# BM25 indexing + retrieval
@dataclass
class BM25Index:
    passages: List[Passage]
    tokenized_corpus: List[List[str]]
    bm25: BM25Okapi


def build_bm25_index(passages: List[Passage]) -> BM25Index:
    tokenized = [tokenize(p.text) for p in passages]
    bm25 = BM25Okapi(tokenized)
    return BM25Index(passages=passages, tokenized_corpus=tokenized, bm25=bm25)


def retrieve(index: BM25Index, query: str, k: int) -> List[int]:

    qtok = tokenize(query)
    scores = index.bm25.get_scores(qtok)
    # scores is a numpy array
    # handle without numpy dependency assumptions
    ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    return ranked[:k]


def build_query_q1(user_turn: str) -> str:
    return user_turn.strip()


def build_query_q2_concat(history: List[Dict[str, Any]], user_turn: str, max_turns: int = 6) -> str:
    parts: List[str] = []
    hist_tail = history[-max_turns:] if max_turns > 0 else history
    for t in hist_tail:
        role = str(t.get("role", "")).strip().lower()
        utt = str(t.get("utterance", "")).strip()
        if utt:
            parts.append(f"{role}: {utt}")
    parts.append(f"user: {user_turn.strip()}")
    return " ".join(parts)


@dataclass
class RecallStats:
    n: int = 0
    doc_hits: int = 0
    span_hits: int = 0

    # gets document-level recall from accumulated stats.
    def doc_recall(self) -> float:
        return self.doc_hits / self.n if self.n else 0.0

    # gets span-level recall from accumulated stats.
    def span_recall(self) -> float:
        return self.span_hits / self.n if self.n else 0.0


def eval_recall(
    index: BM25Index,
    examples: Sequence[Tuple[str, List[Dict[str, Any]], str, str, Set[str], Set[str]]],
    k: int,
    mode: str,
    max_turns_concat: int = 6,
) -> RecallStats:
    stats = RecallStats()
    # precompute passage -> set(span_ids)
    passage_span_sets: List[Set[str]] = [set(p.span_ids) for p in index.passages]

    for ex_id, hist, user_turn, _agent_turn, gold_doc_ids, gold_span_ids in examples:
        if mode == "q1":
            q = build_query_q1(user_turn)
        elif mode == "q2":
            q = build_query_q2_concat(hist, user_turn, max_turns=max_turns_concat)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        top_idx = retrieve(index, q, k=k)
        retrieved_docs = {index.passages[i].doc_id for i in top_idx}
        retrieved_spans: Set[str] = set()
        for i in top_idx:
            retrieved_spans |= passage_span_sets[i]

        # doc-level hit if any gold DOC present
        doc_hit = bool(gold_doc_ids & retrieved_docs) if gold_doc_ids else False
        # span-level hit if any gold SPAN ID present
        span_hit = bool(gold_span_ids & retrieved_spans) if gold_span_ids else False

        stats.n += 1
        stats.doc_hits += int(doc_hit)
        stats.span_hits += int(span_hit)

    return stats


# main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--history_turns", type=int, default=6, help="Turns to keep for building example history.")
    ap.add_argument("--max_eval", type=int, default=500, help="How many examples to use for quick recall sanity eval.")
    ap.add_argument("--min_passage_chars", type=int, default=60, help="Filter out very short sections.")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    docs_json, dial_train_json, dial_val_json = locate_multidoc2dial_files(data_dir)

    print("docs_json:", docs_json)
    print("train_json:", dial_train_json)
    print("val_json:  ", dial_val_json)

    docs_by_domain = load_docs(docs_json)
    total_docs = sum(len(v) for v in docs_by_domain.values())
    print(f"\nLoaded docs: domains={len(docs_by_domain)} total_docs={total_docs}")
    for d, v in list(docs_by_domain.items())[:5]:
        print(f"  - domain={d} docs={len(v)}")

    dials_train_by_domain = load_dialogues(dial_train_json)
    dials_val_by_domain = load_dialogues(dial_val_json)

    n_train = sum(len(v) for v in dials_train_by_domain.values())
    n_val = sum(len(v) for v in dials_val_by_domain.values())
    print(f"\nLoaded dialogues: train={n_train} val={n_val}")
    for d, v in list(dials_train_by_domain.items())[:5]:
        print(f"  - train domain={d} dialogues={len(v)}")

    # get one sample dialogue
    print("\n--- Sample dialogue (first 8 turns) ---")
    sample = None
    for dom, dials in dials_train_by_domain.items():
        if dials:
            sample = dials[0]
            break
    if sample:
        print("dial_id:", sample.get("dial_id"))
        turns = sample.get("turns", [])
        if isinstance(turns, list):
            for t in turns[:8]:
                if not isinstance(t, dict):
                    continue
                role = str(t.get("role", ""))
                utt = str(t.get("utterance", "")).strip()
                refs = t.get("references", [])
                print(f"{role:>6}: {utt[:140]}")
                if refs:
                    r0 = refs[0] if isinstance(refs, list) and refs else None
                    if isinstance(r0, dict):
                        print(f"        refs[0]: doc_id={r0.get('doc_id')} id_sp={r0.get('id_sp')} label={r0.get('label')}")
        else:
            print("sample turns not a list; check format.")
    else:
        print("ISSUE HERE: No sample dialogue found.")

    # for passages
    print("\nBuilding section passages from spans...")
    passages = build_section_passages(docs_by_domain, min_chars=args.min_passage_chars)
    print(f"Built passages: {len(passages)} (min_chars={args.min_passage_chars})")

    # for BM25 index
    print("Building BM25 index...")
    index = build_bm25_index(passages)
    print("BM25 ready.")

    # saving artifacts
    processed_dir = data_dir / "processed"
    indices_dir = data_dir / "indices"
    safe_mkdir(processed_dir)
    safe_mkdir(indices_dir)

    passages_path = processed_dir / "passages.jsonl"
    meta_path = processed_dir / "passage_meta.json"
    bm25_path = indices_dir / "bm25_index.pkl"

    write_jsonl(
        passages_path,
        (
            {
                "passage_id": p.passage_id,
                "domain": p.domain,
                "doc_id": p.doc_id,
                "id_sec": p.id_sec,
                "title_path": p.title_path,
                "text": p.text,
                "span_ids": p.span_ids,
            }
            for p in passages
        ),
    )
    write_json(
        meta_path,
        {
            "passage_count": len(passages),
            "min_passage_chars": args.min_passage_chars,
            "tokenizer": "regex_alnum_lower",
            "notes": "Passages grouped by (doc_id, id_sec) using spans.text_sec; text prefixed with doc title + section titles.",
        },
    )
    with open(bm25_path, "wb") as f:
        pickle.dump(
            {
                "passages": passages,
                "bm25": index.bm25,
                "tokenizer": "regex_alnum_lower",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"\nSaved:\n  {passages_path}\n  {meta_path}\n  {bm25_path}")

    # retrieval eval on a subset of TRAIN dialogues
    all_examples = list(iter_agent_examples(dials_train_by_domain, history_turns=args.history_turns))
    print(f"\nConstructed evaluable (agent-referenced) examples from train: {len(all_examples)}")

    random.shuffle(all_examples)
    eval_subset = all_examples[: min(args.max_eval, len(all_examples))]
    print(f"Evaluating recall on subset n={len(eval_subset)} with k={args.k} ...")

    r_q1 = eval_recall(index, eval_subset, k=args.k, mode="q1")
    r_q2 = eval_recall(index, eval_subset, k=args.k, mode="q2")

    print("\n--- Recall@k sanity check (subset) ---")
    print(f"Q1 (user turn only): doc_recall@{args.k}={r_q1.doc_recall():.3f}  span_recall@{args.k}={r_q1.span_recall():.3f}")
    print(f"Q2 (concat history): doc_recall@{args.k}={r_q2.doc_recall():.3f}  span_recall@{args.k}={r_q2.span_recall():.3f}")


if __name__ == "__main__":
    main()
