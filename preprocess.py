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


def get_synthetic_examples(n: int = 4) -> str:
    """Load a few rows from the synthetic dataset and format as few-shot examples
    for the Gen-Z rewriter prompt.
    Returns a string with examples separated by blank lines.
    """
    import csv, os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    path = os.path.join(DATA_DIR, "genz_synthetic_dataset.csv")
    lines = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                inp = row.get("input_text", "").strip()
                target = row.get("target_text", "").strip()
                if inp and target:
                    lines.append(f"Normal: {inp}\nGenZ: {target}")
    except FileNotFoundError:
        return ""
    return "\n\n".join(lines)

def apply_genz_translation(text: str, reverse_map: dict, force_use: bool = False,
                           intent: str = None, emotion: str = None, tokens: list = None) -> tuple:
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
    
    # 3. Ensure the output uses dataset slang when requested.
    # If the text already contains a known slang term, keep it.
    contains_slang = any(re.search(rf"\b{re.escape(slang)}\b", text, flags=re.IGNORECASE)
                          for slang in _RAW_SLANG)

    slang_source = None
    # Respect emotional safety: don't inject slang into grief/sadness support unless harmless
    if emotion and emotion.lower() in ("sadness", "grief"):
        if force_use and contains_slang:
            print("[SLANG] model output already contained dataset slang; no fallback injection needed")
            slang_source = 'model'
        return text.strip(), slang_source

    if force_use and not contains_slang:
        # Build an intent/emotion-aware allowed set.
        intent = (intent or "").lower()
        emotion = (emotion or "").lower()
        tokens = [t.lower() for t in (tokens or [])]

        general_safe = {"lowkey", "bet", "w", "clutch", "fire", "no cap", "fr", "lit"}
        study_safe   = {"lowkey", "locked in", "clutch", "ngl", "w", "bet"}
        hype_safe    = {"bet", "w", "lit", "fire", "bussin"}
        roast_safe   = {"bet", "fr", "no cap", "deadass"}
        support_safe = {"ngl", "lowkey", "fr"}
        romance_safe = {"rizz", "vibe", "aura"}

        allow = set()
        if intent == "studying":
            allow |= study_safe
        elif intent == "programming":
            allow |= study_safe | general_safe
        elif intent == "emotional_support":
            allow |= support_safe
        elif intent in ("greeting", "jokes"):
            allow |= hype_safe
        elif intent == "roast":
            allow |= roast_safe
        else:
            allow |= general_safe

        romance_tokens = {"crush", "date", "flirt", "rizz", "attraction", "hit on"}
        if any(token in romance_tokens for token in tokens):
            allow |= romance_safe

        # Prefer slang that matches a meaning phrase in the text.
        candidate = None
        meaning_matches = []
        text_lower = text.lower()
        for meaning, slang in reverse_map.items():
            if slang.lower() not in allow:
                continue
            if re.search(r"\b" + re.escape(meaning) + r"\b", text_lower):
                meaning_matches.append((len(meaning), slang))

        if meaning_matches:
            # Choose the most specific meaning match.
            candidate = max(meaning_matches, key=lambda item: item[0])[1]

        # If we still don't have a good candidate, use intent/emotion heuristics.
        if not candidate:
            if intent == "studying":
                for term in ("locked in", "clutch", "lowkey", "ngl", "bet"):
                    if term in allow:
                        candidate = term
                        break
            elif intent in ("greeting", "jokes"):
                for term in ("bet", "w", "lit", "fire"):
                    if term in allow:
                        candidate = term
                        break
            elif intent == "programming":
                for term in ("clutch", "locked in", "lowkey", "bet"):
                    if term in allow:
                        candidate = term
                        break
            elif intent == "emotional_support":
                for term in ("ngl", "lowkey", "fr"):
                    if term in allow:
                        candidate = term
                        break
            elif any(token in romance_tokens for token in tokens):
                for term in ("rizz", "vibe", "aura"):
                    if term in allow:
                        candidate = term
                        break
            elif emotion in ("joy", "surprise"):
                for term in ("fire", "lit", "bussin", "w"):
                    if term in allow:
                        candidate = term
                        break
            elif emotion in ("anger", "fear"):
                for term in ("fr", "ngl", "clutch", "no cap"):
                    if term in allow:
                        candidate = term
                        break

        # Final fallback to a safe dataset slang.
        if not candidate:
            candidate = next((s for s in _RAW_SLANG
                              if s.lower() in allow and s.lower() != "rizz"), None)
        if not candidate:
            candidate = next((s for s in _RAW_SLANG if s.lower() != "rizz"), None)
        if not candidate:
            candidate = random.choice(_RAW_SLANG)

        text = text.rstrip(".?!")
        text = f"{text} — {candidate}"
        print(f"[SLANG] fallback injection: added '{candidate}'")
        slang_source = 'fallback'
    elif force_use and contains_slang:
        print("[SLANG] model output already contained dataset slang; no fallback injection needed")
        slang_source = 'model'

    # 4. Text styling: lowercase and strip trailing periods for that casual Gen Z texting vibe
    text = text.lower().rstrip(".")
    return text.strip(), slang_source




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


def contract_expansions(text: str) -> str:
    """
    Replace expansions with their acronyms using `slang.csv`.
    Example: 'acknowledge' -> 'ack'. Performs longest-match first
    and uses word boundaries to avoid partial replacements.
    """
    if not text:
        return text

    # Build a normalized expansion -> acronym map
    reverse = {}
    for acr, exp in ACRONYM_MAP.items():
        if not exp:
            continue
        norm = re.sub(r"[^\w\s']", " ", exp.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if not norm:
            continue
        # prefer first-seen acronym for a given expansion
        if norm not in reverse:
            reverse[norm] = acr

    # Replace longest expansions first to avoid partial matches
    for expansion in sorted(reverse.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(expansion) + r"\b"
        text = re.sub(pattern, reverse[expansion], text, flags=re.IGNORECASE)

    return text


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

    # Step 5: Replace expansions with acronyms (e.g., 'acknowledge' -> 'ack')
    text = contract_expansions(text)

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


def get_real_slang_for_prompt() -> str:
    """
    Extracts real, usable slang terms from the dataset.
    Filters out generated filler rows like 'genz style expression'
    and 'intensified slang' that add no value to the rewriter.
    Returns a formatted string ready to inject into the LLM prompt.
    """
    import csv, os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    SKIP_MEANINGS = {
        "intensified slang",
        "genz slang expression",
        "strong emphasis",
        "genz style expression",
    }

    seen = set()
    lines = []

    try:
        with open(
            os.path.join(DATA_DIR, "genz_slang_dataset_final2020_2026.csv"),
            encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)
            for row in reader:
                term    = row["slang_term"].strip()
                meaning = row["meaning"].strip().lower()
                example = row.get("example_sentence", "").strip()

                # skip filler rows
                if meaning in SKIP_MEANINGS:
                    continue

                # skip duplicates
                if term.lower() in seen:
                    continue
                seen.add(term.lower())

                lines.append(f'- {term}: "{meaning}" → e.g. "{example}"')

    except FileNotFoundError:
        return "Dataset not found."

    return "\n".join(lines)


def get_emoji_for_context(emotion: str, intent: str, content_keywords: list = None) -> str:
    """
    Dynamically selects the most appropriate emoji from genz_emojis.csv
    based on emotion, intent, and content keywords.
    Returns a single emoji string or empty string if none fits.
    """
    import csv, os, random

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    emoji_data = []
    try:
        with open(os.path.join(DATA_DIR, "genz_emojis.csv"), encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emoji_data.append({
                    "emoji":       row["emoji"].strip(),
                    "name":        row["Name"].strip().lower(),
                    "description": row["Description"].strip().lower(),
                })
    except FileNotFoundError:
        return ""

    if not emoji_data:
        return ""

    candidates = []
    keywords = {k.lower() for k in (content_keywords or [])}

    for row in emoji_data:
        score = 0
        if emotion in ("joy", "surprise") and row["description"]:
            score += 2 if "happy" in row["description"] or "joy" in row["description"] else 0
        if emotion in ("sadness", "anger", "fear") and row["description"]:
            score += 2 if any(term in row["description"] for term in ["sad", "angry", "scared", "heartbroken"]) else 0
        if intent == "studying" and row["description"]:
            score += 2 if "study" in row["description"] or "book" in row["description"] else 0
        if intent == "emotional_support" and row["description"]:
            score += 2 if "hug" in row["description"] or "support" in row["description"] else 0
        if keywords and any(keyword in row["name"] or keyword in row["description"] for keyword in keywords):
            score += 1

        if score > 0:
            candidates.append((score, row["emoji"]))

    if not candidates:
        return ""

    return max(candidates, key=lambda item: item[0])[1]

def get_emoji_for_context(emotion: str, intent: str, content_keywords: list = None) -> str:
    """
    Dynamically selects the most appropriate emoji from genz_emojis.csv
    based on emotion, intent, and content keywords.
    Returns a single emoji string or empty string if none fits.
    """
    import csv, os, random

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # Load emoji dataset
    emoji_data = []
    try:
        with open(os.path.join(DATA_DIR, "genz_emojis.csv"), encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emoji_data.append({
                    "emoji":       row["emoji"].strip(),
                    "name":        row["Name"].strip().lower(),
                    "description": row["Description"].strip().lower(),
                })
    except FileNotFoundError:
        return ""

    # ── Emotion → keyword mapping ──────────────────────────────
    # Maps detected emotion to words we look for in emoji descriptions
    EMOTION_KEYWORDS = {
        "joy":      ["funny", "laugh", "positive", "good", "hype",
                     "stylish", "fire", "lit", "great", "amazing",
                     "celebration", "win", "success"],
        "sadness":  ["sad", "cry", "overwhelm", "heavy", "loss",
                     "tough", "hard", "grief", "miss"],
        "anger":    ["frustrat", "mad", "annoyed", "done", "over"],
        "fear":     ["stress", "nervous", "anxious", "scared",
                     "shock", "overwhelm"],
        "surprise": ["shock", "wow", "unexpected", "surprised",
                     "speechless", "amazed"],
        "neutral":  ["interest", "attention", "curious", "okay",
                     "chill", "vibe", "relax"],
        "love":     ["love", "heart", "care", "warm", "sweet"],
    }

    # ── Intent → keyword mapping ────────────────────────────────
    INTENT_KEYWORDS = {
        "studying":          ["stress", "overwhelm", "work", "focus",
                              "lock", "concentrate", "busy"],
        "programming":       ["work", "focus", "busy", "lock", "grind"],
        "jokes":             ["funny", "laugh", "humor", "haha",
                              "lol", "comedy"],
        "greeting":          ["interest", "attention", "curious",
                              "vibe", "chill"],
        "emotional_support": ["sad", "cry", "heavy", "loss",
                              "tough", "care", "warm"],
        "general":           ["vibe", "chill", "okay", "interest"],
    }

    # ── Content keyword mapping ─────────────────────────────────
    # If specific words appear in the response content
    CONTENT_KEYWORD_MAP = {
        "party":    ["vibe", "chill", "relax", "fun"],
        "exam":     ["stress", "overwhelm", "nervous", "focus"],
        "friend":   ["care", "warm", "interest", "attention"],
        "win":      ["hype", "fire", "lit", "success", "great"],
        "fail":     ["sad", "tough", "loss"],
        "funny":    ["laugh", "funny", "humor"],
        "cool":     ["stylish", "fire", "great", "lit"],
        "help":     ["focus", "attention", "interest"],
        "love":     ["love", "heart", "care", "warm"],
        "tired":    ["overwhelm", "stress", "heavy"],
        "scared":   ["shock", "nervous", "anxious"],
        "angry":    ["frustrated", "mad", "done"],
    }

    # ── Emojis that should NEVER be used ───────────────────────
    BANNED_EMOJIS = {
        "🤡",  # never use clown per dataset rules
        "🧢",  # cap — too easily misread
        "🅱️",  # meme only, no conversational value
        "🅿",   # very niche
        "💳💥💳💥💳💥",  # multi-emoji combo, skip
        "👉👈",           # combo
        "✨✨🧚🧚",       # combo
        "😙👌",           # combo
        "🐸☕",           # combo — too niche
        "🌚🌝",           # combo
    }

    
    # ── Hard banned for study/stress context ──────────────────
    CONTEXT_BANNED = {
        "studying": [
            "🦦",  # relaxing at beach — wrong energy
            "🏖️",  # beach
            "😴",  # sleeping
            "💅",  # nail polish / fancy — wrong context
            "🌜", "🌛",  # moon faces — irrelevant
        ],
        "emotional_support": [
            "💀",  # humor emoji — wrong for grief
            "🔥",  # hype — wrong for sadness
            "💅",  # fancy — wrong context
        ],
        "greeting": [
            "😭",  # overwhelm — too heavy for greeting
            "💀",  # unless genuinely funny
        ],
        "joy": [
            "😭",  # wrong emotion
        ],
    }

    # ── Build candidate search keywords from emotion, intent, and content
    search_keywords = []
    if emotion:
        search_keywords.extend(EMOTION_KEYWORDS.get(emotion, []))
    if intent:
        search_keywords.extend(INTENT_KEYWORDS.get(intent, []))
    if content_keywords:
        search_keywords.extend([kw.lower() for kw in content_keywords if isinstance(kw, str)])

    search_keywords = list(dict.fromkeys(search_keywords))

    banned_for_context = set(BANNED_EMOJIS)
    banned_for_context.update(CONTEXT_BANNED.get(intent, []))
    banned_for_context.update(CONTEXT_BANNED.get(emotion, []))

    # ── Preferred emojis per emotion (from your actual dataset) ──
    EMOTION_PREFERRED = {
        "sadness":  ["😭"],           # "exclusively used for overwhelming feelings"
        "fear":     ["😭", "👀"],     # stress + curiosity
        "joy":      ["🔥", "💀"],     # fire + dead laughing
        "surprise": ["👀", "💀"],     # eyes + disbelief
        "neutral":  ["👀", "🦦"],    # curiosity — 🦦 only for chill neutral
        "anger":    [],               # no emoji for anger
        "love":     ["✨"],           # sparkles
    }

    # Score each emoji
    scored = []
    for item in emoji_data:
        if item["emoji"] in banned_for_context:
            continue

        if " " in item["emoji"] or len(item["emoji"]) > 8:
            continue

        score = 0
        combined = item["name"] + " " + item["description"]

        # Keyword match score
        for keyword in search_keywords:
            if keyword in combined:
                score += 1

        # Bonus for preferred emojis
        if item["emoji"] in EMOTION_PREFERRED.get(emotion, []):
            score += 3

        if score > 0:
            scored.append((item["emoji"], score, item["description"]))

    if not scored:
        # Safe fallbacks per emotion
        SAFE_FALLBACKS = {
            "sadness":  "😭",
            "fear":     "😭",
            "joy":      "🔥",
            "surprise": "👀",
            "neutral":  "👀",
            "anger":    "",
        }
        return SAFE_FALLBACKS.get(emotion, "👀")

    scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored[:3]
    chosen = random.choice(top_candidates)
    return chosen[0]