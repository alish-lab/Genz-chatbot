"""
response_generator.py — LLM-powered Gen Z response generation for GenZBot
────────────────────────────────────────────────────────────────────────────
Architecture (NLP → LLM hybrid):

  NLP pipeline outputs (emotion, sentiment, intent, tokens, emoji emotions)
       ↓
  Prompt engineer a rich, structured context
       ↓
  Anthropic Claude (claude-sonnet-4-20250514) generates the response
       ↓
  Post-process: strip analysis mentions, enforce tone rules
       ↓
  Final Gen Z response

The LLM is NOT doing raw chatbot work — it receives the fully-processed NLP
analysis and uses it to generate a contextually appropriate response.
Grief detection, tone rules, and persona constraints are all enforced via
the system prompt built from the NLP outputs.
────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import random
from dotenv import load_dotenv
from preprocess import _load_slang_map, _build_reverse_slang_map, apply_genz_translation

load_dotenv()

SLANG_MAP = _load_slang_map()
REVERSE_SLANG_MAP = _build_reverse_slang_map(SLANG_MAP)

from groq_service import generate_ai_response
from retrieval import get_top_genz_examples

from preprocess import (
    _load_slang_map,
    _build_reverse_slang_map,
    apply_genz_translation,
    get_real_slang_for_prompt,
    get_emoji_for_context,      # add this
    get_synthetic_examples,
)

# Load once at startup
SLANG_MAP         = _load_slang_map()
REVERSE_SLANG_MAP = _build_reverse_slang_map(SLANG_MAP)
FULL_SLANG_PROMPT = get_real_slang_for_prompt()  # add this

# ─────────────────────────────────────────────────────────────────────────────
# Grief / loss trigger phrases (NLP rule — runs before LLM)
# ─────────────────────────────────────────────────────────────────────────────
GRIEF_PHRASES = [
    "died", "passed away", "lost my cat", "lost my dog", "lost my mom",
    "lost my dad", "lost my friend", "lost my pet", "lost my grandma",
    "lost my grandpa", "funeral", "miss her", "miss him", "miss them",
    "she passed", "he passed", "they passed", "gone forever",
    "can't believe she's gone", "can't believe he's gone",
    "death", "mourning", "grieving", "buried", "cremated",
    "my dog died", "my cat died", "pet died", "someone died",
]

def _detect_grief(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in GRIEF_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback responses (used when API is unavailable)
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK = {
    "grief":             ["I'm so sorry. That kind of loss is heavy and there's no rushing through it. I'm here whenever you need to talk.",
                          "Losing someone you love changes things. Please give yourself the space to feel whatever you're feeling right now."],
    "sadness":           ["hey, I see you. that sounds really hard. want to talk about it?",
                          "ngl that sounds genuinely tough. you're not alone in this fr."],
    "anger":             ["okay I hear you, that sounds genuinely frustrating. what happened?",
                          "fr that sounds annoying. want to vent or do you want advice?"],
    "fear":              ["hey, breathe. that sounds scary but you've got this, I promise.",
                          "I totally get why that's nerve-wracking. let's think through it together."],
    "joy":               ["okay WAIT this actually made my day 🔥 tell me more!!",
                          "the energy you're bringing rn is immaculate ngl ✨"],
    "studying":          ["okay study buddy mode: ACTIVATED 📚 what are we tackling?",
                          "let's get this bread 📖 what subject are we conquering?"],
    "programming":       ["okay I'm basically a rubber duck that can also code 🦆 what's the bug?",
                          "debugging session commencing 💻 what language are we in?"],
    "jokes":             ["why did the scarecrow win an award? because he was outstanding in his field 💀",
                          "fun fact: otters hold hands when they sleep so they don't drift apart 🦦"],
    "greeting":          ["hey!! what's the vibe today? ✨", "yo, what's good? 👀"],
    "general":           ["okay that's actually interesting, go on 👀",
                          "lowkey I have thoughts on this. what's your take?"],
}

SLANG_USAGE_GUIDE = """
SLANG SEMANTIC RULES — only use a term if the meaning fits:

say less    → means "understood, got it, no need to explain"
             ONLY use when acknowledging something the user explained
             NEVER use as a filler or question ending
             WRONG: "what subject is it, say less?"
             RIGHT: "say less, let's get into it"

locked in   → means "fully focused, in the zone"
             use when encouraging focus or study mode
             RIGHT: "let's get locked in fr"

cooked      → means "in trouble, overwhelmed, failed"
             use when situation is bad or difficult
             RIGHT: "we are so cooked for this exam 😭"

lowkey      → means "subtly, quietly, a little bit"
             use to soften an opinion
             RIGHT: "lowkey recursion is actually interesting"

ngl         → means "not gonna lie, being honest"
             use before an honest opinion
             RIGHT: "ngl that quiz sounds rough"

fr          → means "for real, seriously"
             use to emphasize something genuine
             RIGHT: "that's tough fr"

W           → means "win, good outcome"
             use only for genuinely positive things
             RIGHT: "that's actually a W"

no cap      → means "no lie, seriously"
             use to emphasize honesty
             RIGHT: "no cap this is important"

aura        → means "someone's vibe or energy"
             use for personality or vibe descriptions
             RIGHT: "the aura of recursion is just calling itself"

cooked      → means "in trouble or overwhelmed"
             RIGHT: "we're cooked if we don't start now"

bet         → means "okay, sounds good, agreed"
             use as affirmation
             RIGHT: "bet, let's do this"
"""

def _fallback(emotion: str, intent: str, is_grief: bool) -> str:
    if is_grief:
        return random.choice(FALLBACK["grief"])
    if emotion in FALLBACK:
        return random.choice(FALLBACK[emotion])
    if intent in FALLBACK:
        return random.choice(FALLBACK[intent])
    return random.choice(FALLBACK["general"])


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — turns NLP outputs into a structured LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(emotion: str, sentiment: str, intent: str,
                          is_grief: bool, emoji_emotions: list,
                          tokens: list, history: list = None) -> str:
    """
    Build a system prompt that encodes all NLP analysis results as
    explicit behavioral constraints for the LLM.
    """

    # ── Core persona ───────────────────────────────────────────────────────
    persona = """You are an empathetic, helpful AI bestie.
    You were built on a full NLP pipeline: your understanding of this message
    comes from emotion detection, sentiment analysis, intent classification,
    slang normalization, acronym expansion, and emoji interpretation.
    Use that understanding — not surface-level keyword matching — to respond."""

    # ── Tone rules derived from NLP outputs ───────────────────────────────
    tone_rules = []

    if is_grief:
        tone_rules += [
            "The user is experiencing grief or loss. This overrides everything.",
            "DO NOT use Gen Z slang, hype language, or emojis.",
            "Respond like a compassionate, quiet friend — warm, short, human.",
            "Do not try to fix anything. Just acknowledge and be present.",
        ]
    elif emotion == "sadness":
        tone_rules += [
            "The NLP model detected SADNESS. Be comforting and validating.",
            "Use gentle, natural language. Light Gen Z tone is okay but no excessive hype.",
            "Do not minimize their feelings. Make them feel heard.",
        ]
    elif emotion == "fear":
        tone_rules += [
            "The NLP model detected FEAR or ANXIETY. Be reassuring.",
            "Ground them gently. 'You've got this' energy.",
            "Keep it calm and steady.",
        ]
    elif emotion in ("joy", "surprise"):
        tone_rules += [
            "The NLP model detected JOY or positive energy. Match it!",
            "Be enthusiastic, fun, upbeat.",
            "Use standard English. Your response will be translated to slang later.",
        ]
    elif emotion == "anger":
        tone_rules += [
            "The user expressed frustration or anger. Stay calm and SHORT.",
            "Do NOT lecture or moralize.",
            "Do NOT reference previous topics.",
            "1-2 sentences max. Acknowledge and move on.",
        ]
    else:
        tone_rules += [
            "Tone is NEUTRAL. Be conversational and natural.",
            "Adapt based on what the user actually said.",
        ]

    # ── Intent-specific behavior ───────────────────────────────────────────
    intent_rules = []
    if intent == "studying":
        intent_rules += [
            "The user wants STUDY HELP. Become their study buddy.",
            "Be helpful, clear, and encouraging. Use structured explanations.",
            "Prioritize being useful and clear.",
        ]
    elif intent == "programming":
        intent_rules += [
            "The user needs PROGRAMMING HELP. Be technical and helpful.",
            "Ask clarifying questions if needed (language, error, context).",
            "Rubber duck energy — help them think through it.",
        ]
    elif intent == "jokes":
        intent_rules += [
            "The user wants something FUNNY. Deliver a joke, meme reference, or fun fact.",
            "Be genuinely funny, not corny (or be intentionally corny and own it).",
        ]
    elif intent == "greeting":
        intent_rules += [
            "The user is GREETING you. Respond warmly and with good energy.",
            "Ask what's on their mind or how their day is going.",
        ]
    elif intent == "emotional_support":
        intent_rules += [
            "The user needs EMOTIONAL SUPPORT. Be present and empathetic.",
            "Don't rush to solutions. Listen and validate first.",
        ]

    # ── Emoji context from NLP ─────────────────────────────────────────────
    emoji_context = ""
    if emoji_emotions:
        emoji_str = ", ".join(f"{e} ({emo})" for e, emo in emoji_emotions)
        emoji_context = f"\nEmoji signals detected by NLP pipeline: {emoji_str}"
        emoji_context += "\nFactor this emotional context into your response."

    # ── Key tokens from NLP ────────────────────────────────────────────────
    token_context = ""
    if tokens:
        token_context = f"\nKey semantic tokens extracted: {', '.join(tokens[:12])}"

    # ── Hard rules (always enforced) ──────────────────────────────────────
    hard_rules = [
      
        "NEVER mention emotion labels, sentiment scores, or intent classifications.",
        "NEVER say 'I detected that you are...' or 'Based on your sentiment...'",
        "NEVER drag the conversation back to a previous topic the user has moved on from.",
        "NEVER reference recursion, studying, or any prior topic unless the user brings it up.",
        "If the user says something unrelated to the previous topic, just respond to what they said.",
        "NEVER be robotic or formal unless explaining something technical.",
        "Keep responses 1 to 4 sentences max unless explaining something complex.",
        "Write in clear standard English. The response will be styled afterward.",
        "Use 0 to 2 emojis max.",
        "Sound like a real person. Not a tutor. Not a therapist. A smart friend.",
        "If the user expresses frustration or anger toward you, stay calm and brief. Do not lecture.",
        "If the user talks about their social life, friends, or personal life, engage with THAT topic.",
        "NEVER speak as if you ARE the user. You are responding TO the user.",
        "NEVER say 'I have a quiz' or 'I need help' — those are the user's words not yours.",
        "NEVER start a response with 'I' unless you are expressing your own opinion.",
        "NEVER repeat back what the user said as if it is your own situation.",
        "You are an AI assistant responding to the user. Always maintain that perspective.",
    ]
    

    # ── Assemble ──────────────────────────────────────────────────────────
    system = persona + "\n\n"
    system += "NLP Analysis Results:\n"
    system += f"  • Detected emotion  : {emotion}\n"
    system += f"  • Detected sentiment: {sentiment}\n"
    system += f"  • Detected intent   : {intent}\n"
    if emoji_context:
        system += emoji_context + "\n"
    if token_context:
        system += token_context + "\n"
        
    if history:
        system += "\nConversation History (for context):\n"
        for i, msg in enumerate(history[-5:]):
            system += f"  - [{i+1}] {msg}\n"
        system += "\nNote: Factor the above context into your response so you don't repeat yourself or misunderstand follow-ups.\n"

    system += "\n\nTone Rules (derived from NLP analysis):\n"
    for rule in tone_rules:
        system += f"  - {rule}\n"

    if intent_rules:
        system += "\nIntent-specific Rules:\n"
        for rule in intent_rules:
            system += f"  - {rule}\n"

    system += "\nHard Rules (always apply):\n"
    for rule in hard_rules:
        system += f"  - {rule}\n"

    return system

def rewrite_genz(
    groq_response: str,
    intent: str,
    emotion: str,
    retrieved_examples: str,
    selected_emoji: str = "",       # add this
    persona: str = None,
    synthetic_examples: str = "",
) -> str:

    emotion_rules = ""
    if emotion == "sadness":
        emotion_rules = (
            "Emotion is SAD or GRIEF: be human first, drop everything else.\n"
            "NO slang. NO emoji except 😭 if it fits naturally.\n"
            "Short, warm, real. Sound like a friend who actually cares."
        )
    elif emotion == "anger":
        emotion_rules = (
            "Emotion is ANGER: validate first, stay calm.\n"
            "No jokes. No hype. Pick grounding slang only if natural: fr, ngl, deadass."
        )
    elif emotion in ("joy", "surprise"):
        emotion_rules = (
            "Emotion is JOY: match the energy.\n"
            "Pick high-energy slang from the dataset: W, bussin, slay, goated, no cap, fire.\n"
            "Pair with 🔥 or 💀 only if it genuinely fits."
        )
    elif emotion == "fear":
        emotion_rules = (
            "Emotion is STRESS/FEAR: calm and reassuring tone.\n"
            "Pick grounding slang: lowkey, ngl, locked in, fr, clutch.\n"
            "Slightly humorous only if it eases tension naturally."
        )
    else:
        emotion_rules = (
            "Emotion is NEUTRAL: casual smart-friend energy.\n"
            "Pick any slang from the dataset that fits the sentence meaning.\n"
            "Never pick slang just to seem Gen-Z — it must fit naturally."
        )

    intent_rules = ""
    if intent == "greeting":
        intent_rules = (
            "Intent is GREETING: 1 sentence max, punchy opener.\n"
            "Examples of good rhythm:\n"
            "- ayo what's the vibe today 👀\n"
            "- okay bet, what are we on\n"
            "- heyy what's good fr"
        )
    elif intent == "studying":
        intent_rules = (
            "Intent is STUDYING: study buddy energy. Clear explanation first, Gen-Z second.\n"
            "Examples of good rhythm:\n"
            "- ngl recursion is actually a W concept once it clicks\n"
            "- okay so the base case is literally the only exit, no cap\n"
            "- we are so locked in rn 😭 what part is cooked for you\n"
            "- lowkey once you see the pattern it all makes sense fr\n"
            "- the aura of recursion is just calling itself until it's done"
        )
    elif intent == "programming":
        intent_rules = (
            "Intent is PROGRAMMING: explain like a smart friend, not a textbook.\n"
            "Examples of good rhythm:\n"
            "- okay so basically it keeps calling itself — that's its whole aura\n"
            "- ngl this cooked me at first but the base case is the key fr\n"
            "- no cap once you see the pattern it's actually a W\n"
            "- the function is in its main character era — solving itself"
        )
    elif intent == "emotional_support":
        intent_rules = (
            "Intent is EMOTIONAL SUPPORT: present, not performative.\n"
            "NO topic callbacks. NO references to prior conversation.\n"
            "Examples of good rhythm:\n"
            "- nah that's genuinely awful I'm so sorry 😭\n"
            "- losing a pet hits different fr, take your time\n"
            "- ngl that kind of loss is heavy, I'm here\n"
            "- that's really rough no cap, how are you holding up"
        )
    elif intent == "jokes":
        intent_rules = (
            "Intent is JOKES: land it, don't explain it.\n"
            "Examples of good rhythm:\n"
            "- bro really said W and walked out 💀\n"
            "- the aura on that is unmatched no cap\n"
            "- that's bussin fr I won't lie"
        )
    else:
        intent_rules = (
            "Intent is GENERAL: casual conversation, keep it chill.\n"
            "Pick whatever slang fits naturally from the dataset."
        )

    if selected_emoji:
        emoji_instruction = (
            f"SELECTED EMOJI: {selected_emoji}\n"
            f"This emoji was chosen from the Gen-Z emoji dataset based on "
            f"the emotion and content context.\n"
            f"Use it AT MOST ONCE in the response if it fits naturally.\n"
            f"If it does not fit, use NO emoji at all.\n"
            f"NEVER add a different emoji — only use this one or none.\n"
        )
    else:
        emoji_instruction = "Use NO emoji in this response.\n"

    # Persona-specific tweak: roast, hype, vibe
    persona_rules = ""
    persona_name = "Vibe"
    if persona == "roast":
        persona_name = "Roast"
        persona_rules = (
            "PERSONA: Roast — playful, sarcastic, short burns only.\n"
            "Be funny but never cruel: avoid personal attacks and sensitive topics.\n"
            "Use at most one light roast and no emojis unless explicitly requested.\n"
        )
    elif persona == "hype":
        persona_name = "Hype"
        persona_rules = (
            "PERSONA: Hype — energetic, encouraging, high-energy slang allowed.\n"
            "Use 0-2 slang words and at most one emoji to boost energy.\n"
        )
    elif persona == "vibe":
        persona_name = "Vibe"
        persona_rules = (
            "PERSONA: Vibe — chill, supportive, short and atmospheric.\n"
            "Prefer 'lowkey' or 'aura' style slang; minimal emojis.\n"
        )

    prompt = (
        "You are a Gen-Z style rewriter for a university student chatbot.\n\n"
        "YOUR ONLY JOB: rewrite the response below so it sounds like a real "
        "smart uni student texting a close friend.\n\n"

        "HARD RULES:\n"
        "1. Pick ONLY from the APPROVED SLANG DATASET below.\n"
        "   Use AT MOST 1-2 slang terms. Must match sentence meaning.\n"
        f"2. EMOJI RULE:\n{emoji_instruction}\n"
        "HARD RULE: NEVER speak as if you are the user.\n"
        "           Do NOT use first-person statements like 'I', 'I'm', 'I feel', or 'I need' to describe emotions or actions.\n"
        "           Always address the user in second-person (you/your) when referring to their feelings or actions.\n"
        "3. NEVER stack slang: 'ngl lowkey fr no cap' = cringe = fail.\n"
        "4. NEVER reference prior topics during grief or emotional support.\n"
        "5. NEVER use overly literary metaphors.\n"
        "6. Keep it SHORT: 1-3 sentences unless technical.\n"
        "7. Structure: short reaction → message → optional question.\n"
        "8. Clarity before style. Explain cleanly first.\n"
        "9. NEVER mention emotion labels or NLP analysis.\n"
        "10. Sound HUMAN. If it sounds like an AI doing Gen-Z, rewrite it.\n"
        "11. If the rewritten response does not already use approved slang, add one natural dataset term.\n"
        "    Do NOT force slang if it breaks clarity or the emotional tone.\n\n"

        f"EMOTION CONTEXT:\n{emotion_rules}\n\n"
        f"INTENT CONTEXT:\n{intent_rules}\n\n"

        f"SLANG SEMANTIC GUIDE — read before choosing any slang term:\n"
        f"{SLANG_USAGE_GUIDE}\n\n"

        "APPROVED SLANG DATASET — meaning must match:\n"
        f"{FULL_SLANG_PROMPT}\n\n"

        "RETRIEVED STYLE EXAMPLES (rhythm only — never copy):\n"
        f"{retrieved_examples}\n\n" +
        (("FEW-SHOT SYNTHETIC EXAMPLES (NORMAL → GENZ):\n" + synthetic_examples + "\n\n") if synthetic_examples else "") +
        f"{persona_rules}\n"
        "RESPONSE TO REWRITE:\n"
        f"{groq_response}\n\n"
        f"If you sign off, use the persona name: " + persona_name + ".\n\n"
        "OUTPUT ONLY the rewritten response. No labels, no quotes."
    )
    return prompt


def _fix_first_person_response(text: str) -> str:
    """If the LLM replies as if it is the user (starts in first-person),
    convert leading first-person to second-person to avoid role confusion.
    This only applies to leading clauses (very conservative).
    """
    if not text:
        return text

    t = text.lstrip()
    # common contractions and variants
    patterns = [
        (r"\bI['’]?m\b", "you're"),
        (r"\bI'm\b", "you're"),
        (r"\bIm\b", "you're"),
        (r"\bI\b", "you"),
        (r"\bi\b", "you"),
        (r"\bI need\b", "you need"),
        (r"\bI want\b", "you want"),
        (r"\bI should\b", "you should"),
        (r"\bI gotta\b", "you gotta"),
        (r"\bI have to\b", "you have to"),
        (r"\bI am going to\b", "you are going to"),
        (r"\bI’m going to\b", "you're going to"),
        (r"\bI was\b", "you were"),
        (r"\bI feel\b", "you feel"),
    ]
    for pat, repl in patterns:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)

    # As a final catch-all, replace any remaining standalone first-person pronouns
    # that might have been missed (very aggressive safeguard).
    t = re.sub(r"\bI['’]?\b", "you", t)
    t = re.sub(r"\bi\b", "you", t)

    return t.strip() if t != text else text




# ─────────────────────────────────────────────────────────────────────────────
# Groq API call
# ─────────────────────────────────────────────────────────────────────────────

# The LLM API call logic has been refactored into groq_service.py
# which handles caching, debouncing, exponential backoff retries, and errors.


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing — strip any leaked analysis language
# ─────────────────────────────────────────────────────────────────────────────

_LEAKED_PATTERNS = [
    r"(?i)i (detected|noticed|saw|found) (that you|your)",
    r"(?i)based on (your|the) (sentiment|emotion|intent|analysis)",
    r"(?i)(emotion|sentiment|intent)\s*[:=]\s*\w+",
    r"(?i)the (nlp|model|analysis) (detected|found|shows?)",
    r"(?i)according to (my analysis|the analysis|nlp)",
]

def _postprocess(text: str) -> str:
    """Remove any leaked analytical language from LLM output."""
    for pattern in _LEAKED_PATTERNS:
        text = re.sub(pattern, "", text)
    text = re.sub(r"  +", " ", text).strip()
    text = re.sub(r"^[,.\s]+", "", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    message: str,
    emotion: str,
    sentiment: str,
    intent: str,
    persona: str = None,
    history: list = None,
    tokens: list = None,
    emoji_emotions: list = None,
) -> str:

    tokens         = tokens or []
    emoji_emotions = emoji_emotions or []

    # ── 1. Grief check ────────────────────────────────────────
    is_grief = _detect_grief(message)

    # ── 2. Select emoji from dataset based on context ─────────
    selected_emoji = get_emoji_for_context(
        emotion=emotion,
        intent=intent,
        content_keywords=tokens,  # NLP tokens from pipeline
    )

    # ── 3. Build system prompt ────────────────────────────────
    system_prompt = _build_system_prompt(
        emotion        = emotion,
        sentiment      = sentiment,
        intent         = intent,
        is_grief       = is_grief,
        emoji_emotions = emoji_emotions,
        tokens         = tokens,
        history        = history,
    )

    # ── 4. Pass 1: Generate base response ─────────────────────
    normal_response = generate_ai_response(system_prompt, message)

    if not normal_response:
        return _fallback(emotion, intent, is_grief)

    # ── 5. Retrieve style examples ────────────────────────────
    examples_context = get_top_genz_examples(normal_response, top_n=5)
    # few-shot synthetic examples to nudge the rewriter
    synthetic_examples = get_synthetic_examples(n=3)

    # ── 6. Pass 2: Rewrite in Gen-Z style ─────────────────────
    rewrite_prompt_text = rewrite_genz(
        groq_response      = normal_response,
        intent             = intent,
        emotion            = emotion,
        retrieved_examples = examples_context,
        selected_emoji     = selected_emoji,   # pass it in
        persona            = persona,
        synthetic_examples = synthetic_examples,
    )

    genz_response = generate_ai_response(
        "You are a Gen-Z translator. Output only the requested rewritten text.",
        rewrite_prompt_text
    )

    if not genz_response:
        genz_response = normal_response

    # Final safeguard: if the rewrite did not use dataset slang, inject one natural term.
    genz_response, slang_injected = apply_genz_translation(
        genz_response,
        REVERSE_SLANG_MAP,
        force_use=True,
        intent=intent,
        emotion=emotion,
        tokens=tokens,
    )

    # Prevent the bot from replying as if it is the user
    fixed = _fix_first_person_response(genz_response)
    sanitized = False
    if fixed != genz_response:
        print("[SANITIZE] first-person content rewritten to second-person")
        genz_response = fixed
        sanitized = True

    final_text = _postprocess(genz_response)
    meta = {"sanitized": sanitized, "slang_injected": slang_injected}
    return final_text, meta