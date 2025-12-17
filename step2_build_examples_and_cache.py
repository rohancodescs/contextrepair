#!/usr/bin/env python3
"""
step2_build_examples_and_cache.py

Build Task-II generation examples (history + user_turn -> agent response)
and precompute BM25 retrieval for TRAIN using Q2 (concat) with k passages.

Run (recommended first):
  python step2_build_examples_and_cache.py --data_dir ./data --k 5 --limit_train 2000 --resume

Then full:
  python step2_build_examples_and_cache.py --data_dir ./data --k 5 --resume

Outputs:
  data/processed/examples_train.jsonl
  data/processed/examples_val.jsonl
  data/processed/examples_test.jsonl   (if test split exists)
  data/processed/passage_lookup.pkl
  data/processed/retrieved_train_q2_k5.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi  

try:
    from tqdm import tqdm  
except Exception:  
    tqdm = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# IO helper func
# Creates the directory path if missing.
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Loads JSON content from disk.
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Streams dictionaries from a JSONL file.
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# Writes dictionaries to JSONL, one row per line.
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Appends dictionaries to an existing JSONL file.
def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    safe_mkdir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Searches recursively for the first JSON whose name matches any regex.
def find_first_file(root: Path, patterns: List[str]) -> Optional[Path]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for p in root.rglob("*.json"):
        if any(rx.search(p.name) for rx in compiled):
            return p
    return None


# Locates the train/val/test dialogue JSON files under data/raw.
def locate_multidoc2dial_dialogue_files(data_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Expected {raw_dir} to exist. Did you run step0_bootstrap.py?")

    train_json = find_first_file(raw_dir, [r"(multi|mutl)doc2dial.*dial.*train.*\.json$"])
    val_json = find_first_file(raw_dir, [r"(multi|mutl)doc2dial.*dial.*valid.*\.json$", r"(multi|mutl)doc2dial.*dial.*validation.*\.json$"])
    test_json = find_first_file(raw_dir, [r"(multi|mutl)doc2dial.*dial.*test.*\.json$"])

    if not train_json or not val_json:
        raise FileNotFoundError(f"Could not locate train/val dialogue JSON under {raw_dir}.")
    return train_json, val_json, test_json


# Returns obj[key] if that key exists on a dict, else the original object.
def unwrap_if_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return obj


# Loads a dialogue JSON and groups entries by domain.
def load_dialogues_by_domain(dial_json: Path) -> Dict[str, List[Dict[str, Any]]]:

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


# passage helper
# bm25 helper
# Performs the shared regex tokenization.
def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


# Loads passages and returns ids, doc_ids, texts, and span sets.
def load_passages(passages_jsonl: Path) -> Tuple[List[str], List[str], List[str], List[Set[str]]]:

    pids: List[str] = []
    doc_ids: List[str] = []
    texts: List[str] = []
    span_sets: List[Set[str]] = []

    for row in iter_jsonl(passages_jsonl):
        pid = str(row["passage_id"])
        did = str(row["doc_id"])
        text = str(row["text"])
        spans = row.get("span_ids", [])
        if not isinstance(spans, list):
            spans = []
        pids.append(pid)
        doc_ids.append(did)
        texts.append(text)
        span_sets.append(set(str(x) for x in spans))

    if not pids:
        raise ValueError(f"Loaded 0 passages from {passages_jsonl}")
    return pids, doc_ids, texts, span_sets


# Builds a BM25 index over the supplied texts.
def build_bm25(texts: List[str]) -> BM25Okapi:
    tokenized_corpus = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized_corpus)


# Returns the top-k indices and scores from a BM25 score vector.
def topk_indices_and_scores(scores: Any, k: int) -> List[Tuple[int, float]]:

    try:
        import numpy as np  

        if isinstance(scores, np.ndarray):
            k_eff = min(k, scores.shape[0])
            if k_eff <= 0:
                return []
            idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
            idx = idx[np.argsort(-scores[idx])]
            return [(int(i), float(scores[int(i)])) for i in idx]
    except Exception:
        pass

    # fallback
    n = len(scores)
    k_eff = min(k, n)
    ranked = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)[:k_eff]
    return [(i, float(scores[i])) for i in ranked]


# task-II generation examples (from dialouges)
# Builds generation training examples from dialogues with referenced agent turns.
def build_examples_from_dialogues(
    dialogues_by_domain: Dict[str, List[Dict[str, Any]]],
    history_turns: int = 6,
) -> List[Dict[str, Any]]:

    examples: List[Dict[str, Any]] = []

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
                if not isinstance(refs, list) or len(refs) == 0:
                    continue

                # nearest preceding user turn
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

                hist_start = max(0, j - history_turns)
                hist_raw = [x for x in turns[hist_start:j] if isinstance(x, dict)]
                history = [{"role": str(x.get("role", "")).lower(), "utterance": str(x.get("utterance", "")).strip()} for x in hist_raw]

                gold_doc_ids: Set[str] = set()
                gold_span_ids: Set[str] = set()
                for r in refs:
                    if not isinstance(r, dict):
                        continue
                    if r.get("doc_id") is not None:
                        gold_doc_ids.add(str(r["doc_id"]))
                    if r.get("id_sp") is not None:
                        gold_span_ids.add(str(r["id_sp"]))

                if not gold_doc_ids and not gold_span_ids:
                    continue

                ex_id = f"{domain}::{dial_id}::turn{i}"
                examples.append(
                    {
                        "example_id": ex_id,
                        "domain": domain,
                        "dial_id": dial_id,
                        "agent_turn_index": i,
                        "history": history, # max is history_turns turns
                        "user_turn": user_turn, # current user turn
                        "target": agent_turn, # agent response to generate
                        "gold_doc_ids": sorted(gold_doc_ids),
                        "gold_span_ids": sorted(gold_span_ids),
                    }
                )

    return examples


# Builds the Q2 retrieval query by concatenating history and user turn text.
def build_query_q2_concat(history: List[Dict[str, Any]], user_turn: str, max_turns_concat: int = 6) -> str:
    """
    Query baseline: concat last N history turns + current user turn.
    """
    parts: List[str] = []
    hist_tail = history[-max_turns_concat:] if max_turns_concat > 0 else history
    for t in hist_tail:
        role = str(t.get("role", "")).strip().lower()
        utt = str(t.get("utterance", "")).strip()
        if utt:
            parts.append(f"{role}: {utt}")
    parts.append(f"user: {user_turn.strip()}")
    return " ".join(parts)



# retrieval cache before compute
# Loads the example_ids that have already been cached to skip duplicates.
def load_done_example_ids(cache_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not cache_path.exists():
        return done
    for row in iter_jsonl(cache_path):
        ex_id = row.get("example_id")
        if ex_id:
            done.add(str(ex_id))
    return done


# Precomputes BM25 retrieval results for each example and saves them.
def precompute_retrieval_cache(
    *,
    examples: Sequence[Dict[str, Any]],
    bm25: BM25Okapi,
    passage_ids: List[str],
    passage_doc_ids: List[str],
    passage_span_sets: List[Set[str]],
    k: int,
    out_path: Path,
    max_turns_concat: int,
    resume: bool,
    limit: Optional[int],
) -> None:
    safe_mkdir(out_path.parent)

    done_ids: Set[str] = set()
    if resume:
        done_ids = load_done_example_ids(out_path)
        if done_ids:
            print(f"[cache] Resume enabled: found {len(done_ids)} already cached in {out_path.name}")

    iterable = examples
    if limit is not None:
        iterable = examples[:limit]

    use_tqdm = tqdm is not None
    it = tqdm(iterable, desc=f"BM25 cache (k={k})") if use_tqdm else iterable

    written = 0
    doc_hits = 0
    span_hits = 0
    total = 0

    # append
    buffer: List[Dict[str, Any]] = []

    for ex in it: 
        ex_id = str(ex["example_id"])
        if resume and ex_id in done_ids:
            continue

        history = ex["history"]
        user_turn = str(ex["user_turn"])
        gold_docs = set(str(x) for x in ex.get("gold_doc_ids", []))
        gold_spans = set(str(x) for x in ex.get("gold_span_ids", []))

        query_q2 = build_query_q2_concat(history, user_turn, max_turns_concat=max_turns_concat)
        scores = bm25.get_scores(tokenize(query_q2))
        top = topk_indices_and_scores(scores, k)

        retrieved_docs: Set[str] = set()
        retrieved_spans: Set[str] = set()
        retrieved_list: List[Dict[str, Any]] = []

        for idx, sc in top:
            pid = passage_ids[idx]
            did = passage_doc_ids[idx]
            retrieved_docs.add(did)
            retrieved_spans |= passage_span_sets[idx]
            retrieved_list.append({"passage_id": pid, "doc_id": did, "score": float(sc)})

        doc_hit = bool(gold_docs & retrieved_docs) if gold_docs else False
        span_hit = bool(gold_spans & retrieved_spans) if gold_spans else False

        total += 1
        doc_hits += int(doc_hit)
        span_hits += int(span_hit)

        buffer.append(
            {
                "example_id": ex_id,
                "query_q2": query_q2,
                "retrieved": retrieved_list,
                "doc_hit": doc_hit,
                "span_hit": span_hit,
            }
        )

        # flush buffer
        if len(buffer) >= 250:
            append_jsonl(out_path, buffer)
            written += len(buffer)
            buffer = []

    if buffer:
        append_jsonl(out_path, buffer)
        written += len(buffer)

    print(f"\n[cache] Wrote {written} new cache rows to {out_path}")
    if total > 0:
        print(f"[cache] Sanity recall on cached subset: doc_recall@{k}={doc_hits/total:.3f}  span_recall@{k}={span_hits/total:.3f}")
    else:
        print("[cache] Nothing new was cached (maybe everything was already done).")


# Parses CLI arguments and coordinates example creation plus caching.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--history_turns", type=int, default=6, help="History turns stored in the example.")
    ap.add_argument("--max_turns_concat", type=int, default=6, help="Turns used in Q2 concat query.")
    ap.add_argument("--limit_train", type=int, default=0, help="If >0, only process first N train examples (debug).")
    ap.add_argument("--resume", action="store_true", help="Resume caching retrieval if output file exists.")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    processed_dir = data_dir / "processed"
    safe_mkdir(processed_dir)

    # find dialogues
    train_json, val_json, test_json = locate_multidoc2dial_dialogue_files(data_dir)
    print("train_json:", train_json)
    print("val_json:  ", val_json)
    print("test_json: ", test_json if test_json else "(not found)")

    # load them
    train_by_domain = load_dialogues_by_domain(train_json)
    val_by_domain = load_dialogues_by_domain(val_json)
    test_by_domain: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if test_json:
        test_by_domain = load_dialogues_by_domain(test_json)

    print("\nDialogue counts:")
    print("  train:", sum(len(v) for v in train_by_domain.values()))
    print("  val:  ", sum(len(v) for v in val_by_domain.values()))
    if test_by_domain is not None:
        print("  test: ", sum(len(v) for v in test_by_domain.values()))

    # build some examples
    print("\nBuilding Task-II examples (agent turns with references)...")
    ex_train = build_examples_from_dialogues(train_by_domain, history_turns=args.history_turns)
    ex_val = build_examples_from_dialogues(val_by_domain, history_turns=args.history_turns)
    ex_test: List[Dict[str, Any]] = []
    if test_by_domain is not None:
        ex_test = build_examples_from_dialogues(test_by_domain, history_turns=args.history_turns)

    print(f"Examples: train={len(ex_train)} val={len(ex_val)} test={len(ex_test) if test_by_domain is not None else 'N/A'}")

    # then write them out 
    train_out = processed_dir / "examples_train.jsonl"
    val_out = processed_dir / "examples_val.jsonl"
    test_out = processed_dir / "examples_test.jsonl"

    write_jsonl(train_out, ex_train)
    write_jsonl(val_out, ex_val)
    if test_by_domain is not None:
        write_jsonl(test_out, ex_test)

    print(f"\nSaved examples:\n  {train_out}\n  {val_out}")
    if test_by_domain is not None:
        print(f"  {test_out}")

    # load the passages
    passages_path = processed_dir / "passages.jsonl"
    if not passages_path.exists():
        raise FileNotFoundError(f"Expected passages at {passages_path}. Run step1_build_passages_and_bm25.py first.")
    print("\nLoading passages:", passages_path)
    passage_ids, passage_doc_ids, passage_texts, passage_span_sets = load_passages(passages_path)
    print(f"Loaded passages: {len(passage_ids)}")

    # save lookup path for passages
    lookup_path = processed_dir / "passage_lookup.pkl"
    if not lookup_path.exists():
        print("Saving passage_lookup.pkl ...")
        lookup = {
            pid: {"doc_id": did, "text": txt, "span_ids": sorted(list(ss))}
            for pid, did, txt, ss in zip(passage_ids, passage_doc_ids, passage_texts, passage_span_sets)
        }
        with open(lookup_path, "wb") as f:
            pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved:", lookup_path)
    else:
        print("Found existing:", lookup_path)

    # build fresh bm25
    print("\nBuilding BM25 from passages (this is fast)...")
    bm25 = build_bm25(passage_texts)
    print("BM25 ready.")

    # retrieval cache before compute
    cache_path = processed_dir / f"retrieved_train_q2_k{args.k}.jsonl"
    limit = args.limit_train if args.limit_train > 0 else None
    precompute_retrieval_cache(
        examples=ex_train,
        bm25=bm25,
        passage_ids=passage_ids,
        passage_doc_ids=passage_doc_ids,
        passage_span_sets=passage_span_sets,
        k=args.k,
        out_path=cache_path,
        max_turns_concat=args.max_turns_concat,
        resume=args.resume,
        limit=limit,
    )


if __name__ == "__main__":
    main()
