"""
preprocess.py — Full NLP pipeline for GenZBot
Handles: cleaning → emoji interpretation → slang normalization →
         acronym expansion → tokenization → stopword removal → feature extraction
"""

import re
import csv
import os
import string

# ---------------------------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_slang_map() -> dict:
    """Load Gen Z slang dataset → {term: meaning}"""
    slang_map = {}
    path = os.path.join(DATA_DIR, "genz_slang_dataset_final2020_2026.csv")
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = row["slang_term"].strip().lower()
                meaning = row["meaning"].strip()
                if term and meaning:
                    slang_map[term] = meaning
    except FileNotFoundError:
        print(f"[WARN] Slang dataset not found at {path}")
    return slang_map

def _build_reverse_slang_map(slang_map: dict) -> dict:
    """Build a reverse map from meaning -> slang_term for NLP post-processing."""
    reverse_map = {}
    ignore_meanings = {"intensified slang", "genz slang expression", "strong emphasis", "genz style expression"}
    
    for slang, meaning in slang_map.items():
        meaning = meaning.lower()
        if meaning in ignore_meanings:
            continue
        
        meaning = meaning.replace("being ", "").replace("very ", "")
        parts = [p.strip() for p in meaning.split(" or ")]
        for p in parts:
            if len(p) > 2:
                if p not in reverse_map:
                    reverse_map[p] = slang
                    
    return dict(sorted(reverse_map.items(), key=lambda item: len(item[0]), reverse=True))

def _get_dataset_slang_and_emojis() -> tuple:
    """Extract a raw list of slang terms and emojis directly from the datasets for injection."""
    import csv, os, random
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    slang_list = []
    try:
        with open(os.path.join(DATA_DIR, "genz_slang_dataset_final2020_2026.csv"), encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slang_list.append(row["slang_term"].strip())
    except Exception:
        slang_list = ["fr", "deadass", "no cap", "lit", "bet", "bussin", "W"]

    emoji_list = []
    try:
        with open(os.path.join(DATA_DIR, "genz_emojis.csv"), encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emoji_list.append(row["emoji"].strip())
    except Exception:
        emoji_list = ["💀", "😭", "🔥", "✨"]
        
    return slang_list, emoji_list

_RAW_SLANG, _RAW_EMOJIS = _get_dataset_slang_and_emojis()

def apply_genz_translation(text: str, reverse_map: dict) -> str:
    """Scan standard English text and actively inject Gen Z slang/emojis using the datasets."""
    import re, random
    if not text:
        return text
    
    # 1. Reverse Dictionary Replacement (Meaning -> Slang)
    for meaning, slang in reverse_map.items():
        pattern = r'\b' + re.escape(meaning) + r'\b'
        text = re.sub(pattern, slang, text, flags=re.IGNORECASE)
        
    # 2. Text styling: lowercase and strip periods for that casual Gen Z texting vibe
    text = text.lower().replace(".", "")
    
    # 3. Dynamic Slang Injection (force usage of dataset items)
    # Pick a random "intensified slang" term to append
    intensifiers = [s for s in _RAW_SLANG if s.endswith(" af") or s.startswith("so ")]
    if not intensifiers: intensifiers = _RAW_SLANG
    
    inject_slang = random.choice(intensifiers)
    
    # 4. Dynamic Emoji Injection
    inject_emoji = random.choice(_RAW_EMOJIS)
    
    # 5. User-provided Starter Pool
    starters = [
        "yooo 😭",
        "nah cause",
        "aight bet",
        "ayo hold up",
        "lowkey"
    ]
    inject_starter = random.choice(starters)
    
    # Randomly prepend the starter, or append the intensifier
    roll = random.random()
    if roll < 0.25:
        text = f"{inject_starter} {text}"
    elif roll < 0.5:
        text = f"{text} {inject_slang} {inject_emoji}"
    elif roll < 0.75:
        text = f"{inject_starter} {text} {inject_emoji}"
    else:
        # Just return the LLM's text since it's already prompted to be Gen Z
        pass
        
    return text.strip()




def _load_acronym_map() -> dict:
    """Load slang.csv acronym → expansion"""
    acronym_map = {}
    path = os.path.join(DATA_DIR, "slang.csv")
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                acr = row["acronym"].strip().lower()
                exp = row["expansion"].strip()
                if acr and exp:
                    acronym_map[acr] = exp
    except FileNotFoundError:
        print(f"[WARN] Acronym dataset not found at {path}")
    return acronym_map


def _load_emoji_map() -> dict:
    """Load genz_emojis.csv emoji → description"""
    emoji_map = {}
    path = os.path.join(DATA_DIR, "genz_emojis.csv")
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emoji = row["emoji"].strip()
                desc = row["Description"].strip()
                if emoji and desc:
                    emoji_map[emoji] = desc
    except FileNotFoundError:
        print(f"[WARN] Emoji dataset not found at {path}")
    return emoji_map


# Cache at module level
SLANG_MAP = _load_slang_map()
ACRONYM_MAP = _load_acronym_map()
EMOJI_MAP = _load_emoji_map()

# Emotion-relevant emoji shortcuts (for sentiment boosting)
EMOJI_EMOTION = {
    "😭": "sadness",
    "💀": "joy",
    "🔥": "excitement",
    "✨": "positivity",
    "😊": "joy",
    "😢": "sadness",
    "😡": "anger",
    "😤": "anger",
    "😱": "fear",
    "🥹": "joy",
    "😂": "joy",
    "🤣": "joy",
    "😐": "neutral",
    "🙂": "neutral",
    "🫡": "neutral",
    "❤️": "joy",
    "💔": "sadness",
    "🥺": "sadness",
    "🤩": "joy",
    "😴": "neutral",
}

# ---------------------------------------------------------------------------
# 2. Basic NLTK stopwords (inline to avoid NLTK download issues in prod)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up",
    "down","in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don",
    "should","now","d","ll","m","o","re","ve","y","ain","aren","couldn",
    "didn","doesn","hadn","hasn","haven","isn","ma","mightn","mustn",
    "needn","shan","shouldn","wasn","weren","won","wouldn",
}

# ---------------------------------------------------------------------------
# 3. Pipeline steps
# ---------------------------------------------------------------------------

def extract_emojis(text: str) -> list:
    """Return list of (emoji, emotion) tuples found in text."""
    found = []
    for emoji, emotion in EMOJI_EMOTION.items():
        if emoji in text:
            found.append((emoji, emotion))
    return found


def interpret_emojis(text: str) -> str:
    """Replace emojis with their Gen Z description from dataset."""
    for emoji, desc in EMOJI_MAP.items():
        if emoji in text:
            # Replace multi-char emoji combos too
            text = text.replace(emoji, f" {desc.lower()} ")
    return text


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, strip extra punctuation."""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", "", text)
    # Remove punctuation except apostrophe (for contractions)
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(text: str) -> str:
    """
    Replace Gen Z slang terms with their meanings.
    Uses the genz_slang_dataset_final2020_2026.csv.
    Longest-match first to avoid partial replacements.
    """
    # Sort by length descending so multi-word terms match before single words
    sorted_terms = sorted(SLANG_MAP.keys(), key=len, reverse=True)
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        matched = False
        for term in sorted_terms:
            term_tokens = term.split()
            n = len(term_tokens)
            if tokens[i:i+n] == term_tokens:
                result.extend(SLANG_MAP[term].lower().split())
                i += n
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def expand_acronyms(text: str) -> str:
    """
    Replace acronyms with their expansions.
    Uses slang.csv (3357 entries).
    """
    tokens = text.split()
    expanded = []
    for token in tokens:
        clean_tok = token.strip("'")
        if clean_tok in ACRONYM_MAP:
            expanded.append(ACRONYM_MAP[clean_tok])
        else:
            expanded.append(token)
    return " ".join(expanded)


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove common stopwords, keep emotionally relevant words."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# 4. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_text: str) -> dict:
    """
    Run the complete NLP preprocessing pipeline.

    Returns a dict with intermediate results and the final cleaned text
    ready for feature extraction.
    """
    # Step 1: Extract emojis BEFORE cleaning (emojis removed by clean)
    emoji_emotions = extract_emojis(raw_text)

    # Step 2: Emoji interpretation (replace with text description)
    text = interpret_emojis(raw_text)

    # Step 3: Text cleaning
    text = clean_text(text)

    # Step 4: Slang normalization
    text = normalize_slang(text)

    # Step 5: Acronym expansion
    text = expand_acronyms(text)

    # Step 6: Tokenization
    tokens = tokenize(text)

    # Step 7: Stopword removal
    filtered_tokens = remove_stopwords(tokens)

    # Step 8: Final cleaned string for model input
    processed_text = " ".join(filtered_tokens)

    return {
        "original": raw_text,
        "emoji_emotions": emoji_emotions,          # [(emoji, emotion), ...]
        "processed_text": processed_text,          # for TF-IDF / LSTM
        "tokens": filtered_tokens,                 # token list
        "full_cleaned": text,                      # pre-stopword-removal version
    }


# ---------------------------------------------------------------------------
# 5. Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "ngl fr this is mid 💀",
        "idk bro im lowkey scared 😱",
        "omg she has insane rizz 🔥",
        "i lost my cat today i miss her so much",
    ]
    for s in samples:
        result = run_pipeline(s)
        print(f"\nInput   : {result['original']}")
        print(f"Emojis  : {result['emoji_emotions']}")
        print(f"Output  : {result['processed_text']}")
