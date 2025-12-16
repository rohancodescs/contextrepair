#!/usr/bin/env python3
"""
step0_bootstrap.py

Bootstrap script for the project:
- Checks GPU + key imports
- Downloads MultiDoc2Dial zip from official site
- Extracts it
- Locates and loads the core JSON files (docs + dialogue train/validation)
- Prints stats + a few example turns
- Optionally downloads CANARD (Wiki-augmented) via HuggingFace datasets and prints a sample

Run:
  python step0_bootstrap.py --data_dir ./data

Optional:
  python step0_bootstrap.py --data_dir ./data --skip_canard
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import textwrap
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.request import urlopen, Request


MULTIDOC2DIAL_ZIP_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"


# -------------------------
# Utilities
# -------------------------
def hr(msg: str = "", width: int = 88) -> None:
    if msg:
        pad = max(0, width - len(msg) - 2)
        print(f"\n{'=' * 3} {msg} {'=' * pad}")
    else:
        print("\n" + "=" * width)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sizeof_fmt(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def download_with_progress(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    """
    Download `url` to `dst` with a simple progress indicator.
    Uses urllib only (no requests dependency).
    """
    safe_mkdir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[download] Found existing file: {dst} ({sizeof_fmt(dst.stat().st_size)}). Skipping.")
        return

    print(f"[download] Downloading: {url}")
    print(f"[download] -> {dst}")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None

        tmp = dst.with_suffix(dst.suffix + ".part")
        bytes_done = 0
        t0 = time.time()

        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_done += len(chunk)

                if total_bytes:
                    pct = (bytes_done / total_bytes) * 100
                    elapsed = max(1e-6, time.time() - t0)
                    speed = bytes_done / elapsed
                    print(f"\r  {pct:6.2f}%  {sizeof_fmt(bytes_done)}/{sizeof_fmt(total_bytes)}  "
                          f"({sizeof_fmt(speed)}/s)", end="")
                else:
                    elapsed = max(1e-6, time.time() - t0)
                    speed = bytes_done / elapsed
                    print(f"\r  {sizeof_fmt(bytes_done)}  ({sizeof_fmt(speed)}/s)", end="")

        print()  # newline
        tmp.rename(dst)
        print(f"[download] Done: {dst} ({sizeof_fmt(dst.stat().st_size)})")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    safe_mkdir(extract_dir)
    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        print(f"[extract] Found marker {marker}. Skipping extraction.")
        return

    print(f"[extract] Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")
    print("[extract] Done.")


def find_first_file(root: Path, patterns: List[str]) -> Optional[Path]:
    """
    Search `root` recursively for the first file matching any of the given regex patterns.
    Patterns are applied to the filename (not the full path).
    """
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for p in root.rglob("*.json"):
        name = p.name
        if any(rx.search(name) for rx in compiled):
            return p
    return None


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_text(s: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# -------------------------
# Environment checks
# -------------------------
def check_environment() -> None:
    hr("Environment Check")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    missing = []
    # Core libs we will rely on later
    for pkg in ["torch", "transformers", "datasets", "evaluate", "sacrebleu", "spacy", "rank_bm25"]:
        try:
            __import__(pkg)
        except Exception as e:
            missing.append((pkg, str(e)))

    if missing:
        print("\n[WARN] Missing imports:")
        for pkg, err in missing:
            print(f"  - {pkg}: {err}")
        print("\nInstall suggestions (adjust as needed):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  pip install transformers datasets evaluate sacrebleu spacy rank_bm25 tqdm")
        print("  python -m spacy download en_core_web_sm")
    else:
        print("[OK] Core imports look good.")

    # Torch + GPU
    try:
        import torch  # type: ignore

        print(f"\nTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"GPU[{idx}]: {props.name} | VRAM: {total_gb:.2f} GB")
            # small GPU op sanity check
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            y = (x @ x.T).mean()
            _ = y.item()
            print("[OK] CUDA matmul sanity check passed.")
        else:
            print("[WARN] CUDA not available. You can still run CPU, but training will be slow.")
    except Exception as e:
        print(f"[WARN] Torch/CUDA check failed: {e}")

    # spaCy model check (rule rewriter uses this)
    try:
        import spacy  # type: ignore

        try:
            _ = spacy.load("en_core_web_sm")
            print("[OK] spaCy model en_core_web_sm is available.")
        except Exception:
            print("[WARN] spaCy model en_core_web_sm not found.")
            print("  Run: python -m spacy download en_core_web_sm")
    except Exception:
        pass


# -------------------------
# MultiDoc2Dial bootstrap
# -------------------------
@dataclass
class MultiDoc2DialPaths:
    root: Path
    zip_path: Path
    extracted_dir: Path
    docs_json: Path
    dial_train_json: Path
    dial_val_json: Path


def bootstrap_multidoc2dial(data_dir: Path) -> MultiDoc2DialPaths:
    hr("MultiDoc2Dial Download + Verify")

    raw_dir = data_dir / "raw"
    safe_mkdir(raw_dir)

    zip_path = raw_dir / "multidoc2dial.zip"
    extracted_dir = raw_dir / "multidoc2dial"

    download_with_progress(MULTIDOC2DIAL_ZIP_URL, zip_path)
    extract_zip(zip_path, extracted_dir)

    # Robust file finding: handle possible naming typos (mutldoc2dial vs multidoc2dial)
    docs_json = find_first_file(
        extracted_dir,
        patterns=[
            r"(multi|mutl)doc2dial.*_doc\.json$",
            r"(multi|mutl)doc2dial.*doc.*\.json$",
        ],
    )
    dial_train_json = find_first_file(
        extracted_dir,
        patterns=[
            r"(multi|mutl)doc2dial.*dial.*train.*\.json$",
            r"(multi|mutl)doc2dial.*dial_train.*\.json$",
        ],
    )
    dial_val_json = find_first_file(
        extracted_dir,
        patterns=[
            r"(multi|mutl)doc2dial.*dial.*valid.*\.json$",
            r"(multi|mutl)doc2dial.*dial.*val.*\.json$",
            r"(multi|mutl)doc2dial.*dial_validation.*\.json$",
        ],
    )

    if not docs_json or not dial_train_json or not dial_val_json:
        hr("ERROR: Expected files not found")
        print(f"Looked under: {extracted_dir}")
        print(f"docs_json: {docs_json}")
        print(f"dial_train_json: {dial_train_json}")
        print(f"dial_val_json: {dial_val_json}")
        print("\nTip: list extracted files and check naming. If needed, we'll adjust patterns.")
        raise SystemExit(1)

    print(f"[OK] docs_json:      {docs_json}")
    print(f"[OK] dial_train:    {dial_train_json}")
    print(f"[OK] dial_val:      {dial_val_json}")

    return MultiDoc2DialPaths(
        root=data_dir,
        zip_path=zip_path,
        extracted_dir=extracted_dir,
        docs_json=docs_json,
        dial_train_json=dial_train_json,
        dial_val_json=dial_val_json,
    )


def inspect_multidoc2dial(paths: MultiDoc2DialPaths, max_dialogues_print: int = 1) -> None:
    hr("MultiDoc2Dial Inspect")

    docs = load_json(paths.docs_json)
    dial_train = load_json(paths.dial_train_json)
    dial_val = load_json(paths.dial_val_json)

    # docs structure: typically docs[domain][doc_id] = doc_obj
    if isinstance(docs, dict):
        domains = list(docs.keys())
        n_docs = 0
        for d in domains:
            if isinstance(docs[d], dict):
                n_docs += len(docs[d])
        print(f"Docs: domains={len(domains)} total_docs≈{n_docs}")
    else:
        print(f"Docs: unexpected type: {type(docs)}")

    def count_dialogues(dials_obj: Any) -> Tuple[int, int]:
        # returns (num_dialogues, num_turns)
        n_d = 0
        n_t = 0
        if isinstance(dials_obj, dict):
            for dom, dom_val in dials_obj.items():
                # dom_val can be a list or dict
                if isinstance(dom_val, list):
                    n_d += len(dom_val)
                    for dial in dom_val:
                        if isinstance(dial, dict):
                            turns = dial.get("turns", [])
                            n_t += len(turns)
                        elif isinstance(dial, list):
                            # dial itself is a list of turns
                            n_t += len(dial)
                elif isinstance(dom_val, dict):
                    # possibly keyed by dial_id
                    n_d += len(dom_val)
                    for _, dial in dom_val.items():
                        if isinstance(dial, dict):
                            turns = dial.get("turns", [])
                            n_t += len(turns)
                        elif isinstance(dial, list):
                            n_t += len(dial)
        elif isinstance(dials_obj, list):
            n_d = len(dials_obj)
            for dial in dials_obj:
                if isinstance(dial, dict):
                    n_t += len(dial.get("turns", []))
                elif isinstance(dial, list):
                    n_t += len(dial)
        return n_d, n_t

    n_train_d, n_train_t = count_dialogues(dial_train)
    n_val_d, n_val_t = count_dialogues(dial_val)
    print(f"Dialogues train: {n_train_d} | turns: {n_train_t}")
    print(f"Dialogues  val:  {n_val_d} | turns: {n_val_t}")

    # Print one sample dialogue
    def get_first_dialogue(dials_obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(dials_obj, dict):
            for _, dom_val in dials_obj.items():
                if isinstance(dom_val, list) and dom_val:
                    first = dom_val[0]
                    # If first is a dict, return it; if it's a list, skip
                    if isinstance(first, dict):
                        return first
                if isinstance(dom_val, dict) and dom_val:
                    # first value
                    first = next(iter(dom_val.values()))
                    if isinstance(first, dict):
                        return first
        if isinstance(dials_obj, list) and dials_obj:
            first = dials_obj[0]
            if isinstance(first, dict):
                return first
        return None

    sample = get_first_dialogue(dial_train)
    if not sample:
        print("[WARN] Could not find a sample dialogue to print.")
        return
    
    if not isinstance(sample, dict):
        print("[WARN] Sample dialogue is not a dict, skipping print.")
        return

    print("\nSample dialogue id:", sample.get("dial_id", "<no dial_id>"))
    turns = sample.get("turns", [])
    if not isinstance(turns, list):
        print("[WARN] Turns is not a list, skipping turn details.")
        return
    print(f"Turns in sample: {len(turns)}")
    for t in turns[: min(8, len(turns))]:
        if not isinstance(t, dict):
            continue
        role = t.get("role", "?")
        utt = t.get("utterance", "")
        refs = t.get("references", [])
        print(f"  - {role:5s} | {summarize_text(utt)}")
        if refs and isinstance(refs, list):
            # print first reference only (to avoid walls of text)
            r0 = refs[0]
            if isinstance(r0, dict):
                print(f"      refs[0]: doc_id={r0.get('doc_id')} id_sp={r0.get('id_sp')} label={r0.get('label')}")


# -------------------------
# CANARD bootstrap (neural rewriter training)
# -------------------------
def inspect_canard(data_dir: Path) -> None:
    hr("CANARD (Wiki-augmented) Inspect")

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        print(f"[WARN] datasets not available: {e}")
        return

    # This dataset provides fields: History (list[str]), Question, Rewrite
    # It's derived from CANARD and includes train/test splits.
    name = "gaussalgo/Canard_Wiki-augmented"
    print(f"Loading HF dataset: {name}")

    ds_train = load_dataset(name, split="train")
    ds_test = load_dataset(name, split="test")

    print(f"CANARD train rows: {len(ds_train)}")
    print(f"CANARD test  rows: {len(ds_test)}")

    ex = ds_train[0]
    hist = ex.get("History", [])
    q = ex.get("Question", "")
    rw = ex.get("Rewrite", "")
    print("\nSample CANARD example:")
    print("History:")
    for h in hist[:4]:
        print("  -", summarize_text(h, 140))
    if len(hist) > 4:
        print("  ...")
    print("Question:", summarize_text(q, 200))
    print("Rewrite: ", summarize_text(rw, 200))


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="Where to store downloaded/unzipped data.")
    ap.add_argument("--skip_canard", action="store_true", help="Skip CANARD download/inspect step.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    safe_mkdir(data_dir)

    check_environment()

    paths = bootstrap_multidoc2dial(data_dir)
    inspect_multidoc2dial(paths)

    if not args.skip_canard:
        inspect_canard(data_dir)

    hr("Bootstrap Complete")
    print("Next file we'll implement: passage chunking + BM25 index builder (k=5 friendly).")


if __name__ == "__main__":
    main()