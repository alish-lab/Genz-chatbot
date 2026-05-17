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
    elif emotion == "anger":
        tone_rules += [
            "The NLP model detected ANGER. Stay calm and listen-first.",
            "Acknowledge the frustration before anything else.",
            "No jokes. No hype. Just understanding.",
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
        "NEVER mention emotion labels, sentiment scores, or intent classifications in your response.",
        "NEVER say 'I detected that you are...' or 'Based on your sentiment...'",
        "NEVER say 'I saw you mentioned...'",
        "NEVER combine multiple response templates.",
        "NEVER be robotic or formal unless explaining something technical.",
        "Keep responses conversational — 1 to 4 sentences max unless explaining something.",
        "Write in clear, standard English. Do NOT use slang yet. Focus entirely on being helpful and empathetic.",
        "Use emojis MODERATELY — 0 to 2 per response max.",
        "Sound human. Sound like a real person who gets it.",
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
    retrieved_examples: str
) -> str:
    """Builds the Gen-Z style rewrite prompt using retrieved examples and strict rules."""
    
    # ── Emotional matching rules ──
    emotion_rules = ""
    if emotion == "sadness":
        emotion_rules = "Emotion is SAD: Be gentle, supportive, and use fewer jokes."
    elif emotion == "anger":
        emotion_rules = "Emotion is ANGER: Stay calm, validate their frustration, do not match the anger."
    elif emotion in ("joy", "surprise"):
        emotion_rules = "Emotion is JOY: Use high energy and hype them up."
    elif emotion == "fear":
        emotion_rules = "Emotion is STRESS/FEAR: Be reassuring and grounded."
        
    # ── Intent matching rules ──
    intent_rules = ""
    if intent == "greeting":
        intent_rules = (
            "Intent is GREETING: Keep it extremely short.\n"
            "Examples:\n"
            "User: hi -> Bot: yooo what's up 👋\n"
            "User: hello -> Bot: heyy what we on today\n"
            "User: hey -> Bot: ayo hi 😭"
        )
    elif intent == "studying":
        intent_rules = (
            "Intent is STUDYING: Become a study buddy.\n"
            "Examples:\n"
            "User: quiz tomorrow -> Bot: quiz tomorrow? 😭 what subject are we fighting\n"
            "User: exam stress -> Bot: aight pause 😭 what topic got you stressed\n"
            "User: quiz tomorrow -> Bot: nah quizzes always spawn at the worst time 😭"
        )

    prompt = (
        "Rewrite the following response in authentic Gen-Z internet style.\n\n"
        "STRICT RULES:\n"
        "1. Gen-Z is a STYLE, not random word replacement. Avoid therapist energy.\n"
        "2. Limit slang: Maximum 1 slang phrase per response (e.g. fr, lowkey, lock in, cooked, wild). NEVER stack many together.\n"
        "3. Limit emoji: Maximum 1 emoji per response (Allowed: 😭 🔥 ✋ 💀 👀 👋). NEVER use the clown emoji (🤡). Only use if emotion matches.\n"
        "4. NEVER say 'it's so great to hear from you', 'freaking out for u', or sound like a motivational poster.\n"
        "5. NEVER overuse 'lowkey', 'nah cause', or 'fam'.\n"
        "6. NEVER explain detected emotions.\n"
        "7. Make responses short, texting-like, natural, with university student energy.\n\n"
        f"EMOTION CONTEXT:\n{emotion_rules}\n\n"
        f"INTENT CONTEXT:\n{intent_rules}\n\n"
        "Here is some Gen-Z slang context retrieved from our knowledge base. "
        "Use it as INSPIRATION ONLY — never copy the exact wording blindly:\n\n"
        f"{retrieved_examples}\n\n"
        "Text to rewrite:\n"
        f"{groq_response}\n\n"
        "Output ONLY the rewritten Gen-Z response, nothing else."
    )
    return prompt





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
    history: list = None,
    tokens: list = None,
    emoji_emotions: list = None,
) -> str:
    """
    Generate a response using NLP analysis + Anthropic LLM.

    Flow:
      1. Grief detection (NLP rule, runs before LLM)
      2. Build structured prompt from all NLP outputs
      3. Call Anthropic LLM with that prompt
      4. Post-process to strip leaked analysis language
      5. Fallback to rule-based responses if API unavailable
    """
    tokens         = tokens or []
    emoji_emotions = emoji_emotions or []

    # ── 1. Grief check (NLP rule — highest priority) ──────────────────────
    is_grief = _detect_grief(message)
    if is_grief:
        # We handle grief via LLM prompt rules inside _build_system_prompt
        pass
        
    # ── 1.5 Greeting Intent Bypass ─────────────────────────────────────────
    if intent == "greeting":
        import random
        greetings = [
            "yooo what's up 👋",
            "heyy how's it going",
            "ayo what we on today 😭"
        ]
        return random.choice(greetings)

    # ── 2. Build system prompt from NLP outputs ───────────────────────────
    system_prompt = _build_system_prompt(
        emotion        = emotion,
        sentiment      = sentiment,
        intent         = intent,
        is_grief       = is_grief,
        emoji_emotions = emoji_emotions,
        tokens         = tokens,
        history        = history,
    )

    # ── 3. Pass 1: Generate Normal Response ───────────────────────────────
    normal_response = generate_ai_response(system_prompt, message)
    
    if not normal_response:
        return _fallback(emotion, intent, is_grief)
        
    # ── 4. Retrieve Context Examples ──────────────────────────────────────
    examples_context = get_top_genz_examples(normal_response, top_n=5)
    
    # ── 5. Pass 2: Rewrite in Gen-Z Style ─────────────────────────────────
    # The rewrite_genz function acts as the system prompt for the second LLM call,
    # but it also includes the text to rewrite directly inside it. So we pass it as the user message.
    rewrite_prompt_text = rewrite_genz(
        groq_response=normal_response,
        intent=intent,
        emotion=emotion,
        retrieved_examples=examples_context
    )
    
    genz_response = generate_ai_response("You are a Gen Z translator. Output only the requested rewritten text.", rewrite_prompt_text)
    
    if not genz_response:
        genz_response = normal_response # Fallback to normal if rewrite fails
        
    # ── 6. Cleanup & Final Post-process ───────────────────────────────────
    clean_text = _postprocess(genz_response)
    
    # We no longer apply the manual slang translator (apply_genz_translation)
    # because the RAG LLM pipeline now natively handles style transfer with strict limits.
    return clean_text

    # ── 5. Fallback (no API key or network error) ─────────────────────────
    return _fallback(emotion, intent, is_grief)
