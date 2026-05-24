

import os
import re
import random
import csv
from dotenv import load_dotenv
from groq_service import generate_ai_response
from retrieval import get_top_genz_examples

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─────────────────────────────────────────────────────────────────────────────
# Emoji safety — enforced in post-processing. Load full emoji dataset and
# derive allowed set dynamically rather than using a small hardcoded set.
# ─────────────────────────────────────────────────────────────────────────────
BANNED_EMOJIS  = {
    "🤬", "🤢", "🤮", "😰", "😨", "😧", "😦", "😧",
    "🤰","🦦", "🧢", "💳", "🐸", "🌚", "🌝", 
    "⏳", "🤪", "🙃", "👁️", "🧍", "🅱️", "🎷", "🦟", "🦗",
}


def _load_emoji_map() -> dict:
    """Load genz_emojis.csv -> {emoji: description}."""
    path = os.path.join(DATA_DIR, "genz_emojis.csv")
    emap = {}
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Accept common header names
            for row in reader:
                # try several common column names
                emoji = row.get("emoji") or row.get("Emoji")
                desc = row.get("description") or row.get("Description") or row.get("desc")
                if emoji:
                    emoji = emoji.strip()
                    desc = (desc or "").strip()
                    if emoji and emoji not in BANNED_EMOJIS:
                        emap[emoji] = desc
    except FileNotFoundError:
        # Fallback small set
        emap = {"😭": "crying", "🔥": "fire", "💀": "dead", "👀": "eyes"}
    return emap


EMOJI_MAP = _load_emoji_map()
ALLOWED_EMOJIS = set(EMOJI_MAP.keys())


def _load_slang_map() -> dict:
    """Load genz_slang_dataset_final2020_2026.csv -> {slang: meaning}"""
    path = os.path.join(DATA_DIR, "genz_slang_dataset_final2020_2026.csv")
    smap = {}
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slang = (row.get("slang_term") or row.get("slang") or "").strip()
                meaning = (row.get("meaning") or row.get("definition") or "").strip()
                if slang and meaning:
                    smap[slang] = meaning
    except FileNotFoundError:
        smap = {}
    return smap


SLANG_MAP = _load_slang_map()

from difflib import SequenceMatcher


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _pick_emoji(message: str, emotion: str, tokens: list, emoji_emotions: list) -> str:
    """Pick best emoji from EMOJI_MAP matching emotion, tokens or message content."""
    message = (message or "").lower()
    candidates = []
    # prefer emojis explicitly linked to detected emoji_emotions
    for e, emo in (emoji_emotions or []):
        if e in EMOJI_MAP and e not in BANNED_EMOJIS:
            return e

    # score emojis by description match against emotion or message
    for emoji, desc in EMOJI_MAP.items():
        if emoji in BANNED_EMOJIS:
            continue
        desc_l = (desc or "").lower()
        score = 0.0
        if emotion and emotion.lower() in desc_l:
            score += 0.6
        # token overlap
        for t in (tokens or []):
            if t and t.lower() in desc_l:
                score += 0.15
        # similarity to message
        score += 0.25 * _similarity(desc_l, message)
        candidates.append((score, emoji))

    candidates.sort(reverse=True)
    if not candidates:
        return None
    top_score, top_emoji = candidates[0]
    if top_score > 0.25:
        return top_emoji
    return None


def _load_acronym_map() -> dict:
    """Load slang.csv acronym -> expansion, then invert to expansion -> acronym."""
    path = os.path.join(DATA_DIR, "slang.csv")
    acronym_map = {}
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                acr = (row.get("acronym") or "").strip().lower()
                exp = (row.get("expansion") or "").strip().lower()
                if acr and exp:
                    acronym_map[exp] = acr
    except FileNotFoundError:
        acronym_map = {}
    return acronym_map

ACRONYM_MAP = _load_acronym_map()


def _pick_slang(message: str, response: str, tokens: list) -> str:
    """Return a slang term from SLANG_MAP whose meaning best matches the response/message.
    Conservative: require minimum similarity threshold before choosing.
    """
    m = (message or "").lower()
    r = (response or "").lower()
    best = (0.0, None)
    for slang, meaning in SLANG_MAP.items():
        meaning_l = meaning.lower()
        # prefer meanings that appear as substrings
        if meaning_l and (meaning_l in r or meaning_l in m):
            return slang
        # otherwise use similarity against response
        sim = _similarity(meaning_l, r)
        if sim > best[0]:
            best = (sim, slang)
    if best[0] > 0.45:
        return best[1]
    return None


def _apply_acronym_expansions(text: str) -> str:
    """Replace longer expansion phrases in the text with known acronyms."""
    if not ACRONYM_MAP:
        return text
    lowered = text.lower()
    # prioritize longer expansions to avoid partial matches
    expansions = sorted(ACRONYM_MAP.keys(), key=len, reverse=True)
    for exp in expansions:
        if exp in lowered:
            acronym = ACRONYM_MAP[exp]
            text = re.sub(rf"\b{re.escape(exp)}\b", acronym, text, flags=re.IGNORECASE)
            lowered = text.lower()
    return text


def _sanitize_emojis(text: str) -> str:
    # strip banned emojis
    for emoji in BANNED_EMOJIS:
        text = text.replace(emoji, "")
    # keep only emojis that exist in our dataset
    found = [e for e in ALLOWED_EMOJIS if e in text]
    if len(found) > 1:
        for extra in found[1:]:
            text = text.replace(extra, "")
    return text.strip()


def _limit_slang(text: str) -> str:
    if not SLANG_MAP:
        return re.sub(r"  +", " ", text).strip()
    lower = text.lower()
    found = [slang for slang in SLANG_MAP.keys() if slang in lower]
    if len(found) > 1:
        # Keep the most specific / longest slang phrase and remove the rest.
        found.sort(key=len, reverse=True)
        keep = found[0]
        for extra in found[1:]:
            text = re.sub(rf"\b{re.escape(extra)}\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", text).strip()


def _postprocess(text: str) -> str:
    leaked = [
        r"(?i)i (detected|noticed|saw|found) (that you|your)",
        r"(?i)based on (your|the) (sentiment|emotion|intent|analysis)",
        r"(?i)the (nlp|model|analysis) (detected|found|shows?)",
    ]
    for p in leaked:
        text = re.sub(p, "", text)
    text = re.sub(r"\*\*?(.+?)\*\*?", r"\1", text)
    return re.sub(r"  +", " ", text).strip().lstrip(".,")


# ─────────────────────────────────────────────────────────────────────────────
# Grief detection
# ─────────────────────────────────────────────────────────────────────────────
GRIEF_PHRASES = [
    "died", "passed away", "lost my cat", "lost my dog", "lost my mom",
    "lost my dad", "lost my pet", "lost my grandma", "lost my grandpa",
    "funeral", "she passed", "he passed", "they passed", "death",
    "mourning", "grieving", "my dog died", "my cat died",
]

def _detect_grief(message: str) -> bool:
    return any(p in message.lower() for p in GRIEF_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# Fallbacks (used when API is unavailable / rate limited)
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK = {
    "grief":    [
        "i'm so sorry. that kind of loss is really heavy. i'm here if you want to talk.",
        "losing someone you love changes everything. take all the time you need.",
    ],
    "sadness":  [
        "hey, that sounds really hard 😭 want to talk about it?",
        "ngl that sounds tough. you're not alone fr.",
    ],
    "anger":    [
        "okay that sounds genuinely frustrating. what happened?",
        "fr that's annoying. want to vent or nah?",
    ],
    "fear":     [
        "hey breathe — you've got this 👀",
        "that's nerve-wracking but let's think through it together.",
    ],
    "joy":      [
        "okay that's actually amazing 🔥 tell me more!!",
        "the energy you're bringing rn is immaculate ngl",
    ],
    "studying": [
        "okay study buddy mode activated — what are we tackling?",
        "let's lock in — what subject are we fighting?",
    ],
    "jokes":    [
        "why did the scarecrow win an award? outstanding in his field 💀",
        "fun fact: a group of flamingos is called a flamboyance. you're welcome.",
    ],
    "greeting": ["heyy what's the vibe", "yo what's good 👀"],
    "general":  [
        "okay that's actually interesting, go on 👀",
        "lowkey have thoughts on this — what's your take?",
    ],
}

def _fallback(emotion: str, intent: str, is_grief: bool) -> str:
    if is_grief:
        return random.choice(FALLBACK["grief"])
    return random.choice(
        FALLBACK.get(emotion) or FALLBACK.get(intent) or FALLBACK["general"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tone and intent lookup tables (replaces long if/elif chains)
# ─────────────────────────────────────────────────────────────────────────────
_TONE = {
    "sadness":  "comforting and gentle — no hype or jokes",
    "anger":    "calm and listen-first — acknowledge frustration first",
    "fear":     "reassuring and grounded — 'you've got this' energy",
    "joy":      "enthusiastic and fun — match their energy",
    "surprise": "enthusiastic and fun — match their energy",
    "neutral":  "chill and conversational",
}

_INTENT_HINT = {
    "studying":          "be a study buddy — helpful and encouraging. Ask what topic ONLY if not already in history.",
    "programming":       "be a debug buddy — technical and helpful. Ask language/error only if not given.",
    "jokes":             "deliver ONE short joke or fun fact then stop — no rambling after the punchline.",
    "greeting":          "be warm and brief — 1 sentence, ask what's on their mind.",
    "emotional_support": "be present and empathetic — validate first, no rushing to solutions.",
    "general":           "engage with what they said directly — respond to content, don't just ask follow-ups.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Compact single-pass prompt builder (~300 tokens target)
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    message: str,
    emotion: str,
    intent: str,
    is_grief: bool,
    history: list,
    examples: str,
) -> tuple:
    """Returns (system_prompt, user_message)."""

    if is_grief:
        system = (
            "You are a compassionate, quiet friend. "
            "The user is grieving. Respond warmly and briefly in 2 sentences max. "
            "No slang. No emojis. Just be present and human."
        )
        return system, message

    tone     = _TONE.get(emotion, "chill and conversational")
    hint     = _INTENT_HINT.get(intent, "be helpful and conversational")

    # History — only last 3 messages, compact format
    history_block = ""
    if len(history) > 1:
        recent = history[-3:]
        history_block = "History: " + " → ".join(f'"{m}"' for m in recent) + "\n"

    # Examples — only first 2, trimmed
    examples_block = ""
    if examples and "No" not in examples:
        lines = [l for l in examples.strip().split("\n\n") if l.strip()][:2]
        if lines:
            examples_block = "Style refs:\n" + "\n".join(lines) + "\n"

    # We do not enumerate the entire emoji set in the prompt to avoid huge prompts.
    # Instruct the model to pick at most one emoji from the dataset (dataset-driven),
    # and explicitly forbid the banned set.
    allowed = "one emoji from the dataset (avoid banned emojis)"

    system = (
        "You are GenZBot — an empathetic AI bestie who texts like a Gen Z university student.\n"
        f"emotion={emotion} | intent={intent} | tone: {tone}\n"
        f"task: {hint}\n"
        f"{history_block}"
        f"{examples_block}"
        "RULES:\n"
        "• 1-3 sentences, lowercase, no ending periods\n"
        f"• MAX 1 slang term from the dataset OR none\n"
        f"• MAX 1 emoji: choose at most one emoji from the project's emoji dataset — or none. NEVER use banned or unlisted emojis\n"
        "• NEVER stack slang (e.g. 'locked in no cap say less' = violation)\n"
        "• NEVER mention emotions/NLP/analysis\n"
        "• NEVER invent context not in the message\n"
        "• NEVER repeat a question already asked in history\n"
        "• Sound like a real person, not a motivational poster\n"
    )

    return system, message


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    message: str,
    emotion: str,
    sentiment: str,
    intent: str,
    history: list = None,
    tokens: list = None,
    emoji_emotions: list = None,
) -> str:
    history        = history or []
    tokens         = tokens or []
    emoji_emotions = emoji_emotions or []

    # ── 1. Grief check ────────────────────────────────────────────────────────
    is_grief = _detect_grief(message)

    # ── 2. Greeting bypass — no API call needed ───────────────────────────────
    if intent == "greeting" and not is_grief:
        return random.choice([
            "sup bestie👋",
            "Hey Buddy👋",
            "yo yo yo! what's poppin",
            "Hey Dude"
            "heyy what's the vibe",
            "yo what's good 👀",
            "hey!! what we on today",
            "ayo what's up 😭",

        ])

    # ── 3. RAG — retrieve style examples ─────────────────────────────────────
    examples = ""
    if not is_grief:
        examples = get_top_genz_examples(message, top_n=3)

    # ── 4. Build compact single-pass prompt ───────────────────────────────────
    system_prompt, user_msg = _build_prompt(
        message  = message,
        emotion  = emotion,
        intent   = intent,
        is_grief = is_grief,
        history  = history,
        examples = examples,
    )

    # ── 5. Single API call ────────────────────────────────────────────────────
    print(f"[GEN] intent={intent} emotion={emotion} grief={is_grief}")
    response = generate_ai_response(system_prompt, user_msg)
    print(f"[GEN] {'OK: ' + repr(response[:60]) if response else 'FAILED — using fallback'}")

    if not response:
        return _fallback(emotion, intent, is_grief)

    # ── 6. Post-processing ────────────────────────────────────────────────────
    clean = _postprocess(response)

    if not is_grief:
        clean = _limit_slang(clean)
        clean = _apply_acronym_expansions(clean)
        clean = _sanitize_emojis(clean)

    # Attempt dataset-driven emoji selection and slang injection
    chosen_emoji = None
    if not is_grief:
        chosen_emoji = _pick_emoji(message, emotion, tokens, emoji_emotions)

    # If intent is jokes and emoji matches 'laugh' or 'fun', allow single-emoji reply
    if chosen_emoji and intent == "jokes":
        desc = (EMOJI_MAP.get(chosen_emoji) or "").lower()
        if any(k in desc for k in ("laugh", "lol", "fun", "haha", "joy", "happy")):
            return chosen_emoji

    # If response is empty or very short, but emoji is strong match, return emoji-only
    if chosen_emoji and (len(clean) <= 8 or not clean.strip()):
        return chosen_emoji

    # If no emoji present in generated text, append the chosen emoji
    if chosen_emoji and not any(e in clean for e in ALLOWED_EMOJIS):
        clean = f"{clean} {chosen_emoji}".strip()

    # Slang injection: replace or append a single slang term if meaningful
    if SLANG_MAP:
        slang_choice = _pick_slang(message, clean, tokens)
        if slang_choice:
            # Append the slang term if not already present
            if slang_choice not in clean.lower():
                clean = f"{clean} {slang_choice}".strip()

    return clean
