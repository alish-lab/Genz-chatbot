"""
retrieval.py — TF-IDF Gen Z example retrieval for GenZBot
───────────────────────────────────────────────────────────
Loads two datasets:
  1. genz_slang_dataset_final2020_2026.csv  — original slang definitions
  2. genz_synthetic_dataset.csv             — synthetic conversational examples
     (columns: text, translation, tags, platform, length)

Fits a single TF-IDF vectorizer over both corpora combined.
The synthetic dataset is weighted more heavily in retrieval because
its conversational examples are much closer to actual bot output style.
"""

import os
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")

SLANG_PATH     = os.path.join(DATA_DIR, "genz_slang_dataset_final2020_2026.csv")
SYNTHETIC_PATH = os.path.join(DATA_DIR, "genz_synthetic_dataset.csv")

# Raised from 0.05 → 0.10 to avoid injecting loosely related examples
CONFIDENCE_THRESHOLD = 0.10

_vectorizer   = None
_tfidf_matrix = None
_dataset      = []          # list of {type, display} dicts


# Slang meanings to skip — too generic to be useful style examples
_SKIP_MEANINGS = {
    "intensified slang", "genz slang expression",
    "strong emphasis", "genz style expression",
}


def _load_slang_dataset() -> list:
    """Load original slang CSV → list of corpus strings + display dicts."""
    entries = []
    try:
        with open(SLANG_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meaning = row.get("meaning", "").strip()
                example = row.get("example_sentence", "").strip()
                slang   = row.get("slang_term", "").strip()

                if meaning.lower() in _SKIP_MEANINGS:
                    continue
                if not meaning or not example:
                    continue

                entries.append({
                    "corpus":  f"{meaning} {example}",
                    "display": f'"{slang}" — {meaning}\nExample: {example}',
                    "source":  "slang",
                })
    except FileNotFoundError:
        print(f"[WARN] Slang dataset not found: {SLANG_PATH}")
    return entries


def _load_synthetic_dataset() -> list:
    """
    Load synthetic Gen Z conversational dataset.
    Expected columns: text, translation, tags, platform, length
    Falls back gracefully if file or columns differ.
    """
    entries = []
    try:
        with open(SYNTHETIC_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # Support multiple possible column name formats
                text        = (row.get("text") or row.get("genz_text") or "").strip()
                translation = (row.get("translation") or row.get("standard_english") or "").strip()
                tags        = (row.get("tags") or row.get("category") or "").strip()

                if not text or not translation:
                    continue

                # Corpus = translation (standard English) so TF-IDF matches
                # on meaning, then we return the Gen Z text as the style example
                corpus_text = f"{translation} {tags}"

                entries.append({
                    "corpus":  corpus_text,
                    "display": f'Gen Z: "{text}"\nMeans: {translation}',
                    "source":  "synthetic",
                })

        print(f"[INFO] Loaded {len(entries)} synthetic Gen Z examples.")
    except FileNotFoundError:
        print(f"[INFO] Synthetic dataset not found at {SYNTHETIC_PATH} — using slang dataset only.")
    except Exception as e:
        print(f"[WARN] Error loading synthetic dataset: {e}")
    return entries


def _init_retrieval_module():
    """Load both datasets and fit TF-IDF (runs once, cached in module globals)."""
    global _vectorizer, _tfidf_matrix, _dataset

    if _vectorizer is not None:
        return

    slang_entries     = _load_slang_dataset()
    synthetic_entries = _load_synthetic_dataset()

    # Combine — synthetic entries added twice to give them more weight
    # (simple weighting without needing separate vectorizers)
    _dataset = slang_entries + synthetic_entries + synthetic_entries

    corpus = [e["corpus"] for e in _dataset]

    if not corpus:
        print("[ERROR] No retrieval data available.")
        return

    try:
        _vectorizer   = TfidfVectorizer(
            stop_words  = "english",
            max_features = 3000,
            ngram_range  = (1, 2),
        )
        _tfidf_matrix = _vectorizer.fit_transform(corpus)
        print(f"[INFO] Retrieval module ready — {len(_dataset)} entries ({len(slang_entries)} slang + {len(synthetic_entries)} synthetic).")
    except Exception as e:
        print(f"[ERROR] TF-IDF fit failed: {e}")


def get_top_genz_examples(
    query: str,
    top_n: int = 3,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> str:
    """
    Retrieve top N Gen Z style examples most relevant to the query.
    Returns a formatted string for injection into the prompt.
    """
    _init_retrieval_module()

    if not _vectorizer or not _dataset or not query:
        return "No context examples available."

    try:
        query_vec    = _vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
        top_indices  = np.argsort(similarities)[-top_n:][::-1]

        seen     = set()
        examples = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score < confidence_threshold:
                continue

            display = _dataset[idx]["display"]
            # Deduplicate identical examples (synthetic entries appear twice)
            if display in seen:
                continue
            seen.add(display)
            examples.append(display)

        return "\n\n".join(examples) if examples else "No relevant examples found."

    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return "No context examples available."
