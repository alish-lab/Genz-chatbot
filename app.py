"""
app.py — GenZBot Flask backend
────────────────────────────────────────────────────────────────
Routes:
  GET  /          → chat UI
  POST /api/chat  → { message } → { response }
────────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
load_dotenv()

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────────────────────────────────────
# Model globals
# ─────────────────────────────────────────────────────────────────────────────
emotion_model      = None
emotion_tokenizer  = None
emotion_le         = None

sentiment_model     = None
sentiment_tokenizer = None
sentiment_le        = None

intent_vectorizer  = None
intent_clf         = None
intent_le          = None

MAX_LEN = 50


def _path(filename: str) -> str:
    return os.path.join(MODEL_DIR, filename)


def load_models():
    global emotion_model, emotion_tokenizer, emotion_le
    global sentiment_model, sentiment_tokenizer, sentiment_le
    global intent_vectorizer, intent_clf, intent_le

    # ── Emotion LSTM ──────────────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model
        emotion_model_path = _path("emotion_rnn_model.h5")
        emotion_tokenizer_path = _path("emotion_tokenizer.joblib")
        emotion_label_path = _path("emotion_label_encoder.joblib")
        if not (os.path.exists(emotion_model_path) and os.path.exists(emotion_tokenizer_path) and os.path.exists(emotion_label_path)):
            missing = [p for p in [emotion_model_path, emotion_tokenizer_path, emotion_label_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Emotion model assets missing: {', '.join(missing)}")
        emotion_model     = load_model(emotion_model_path, compile=False)
        emotion_tokenizer = joblib.load(emotion_tokenizer_path)
        emotion_le        = joblib.load(emotion_label_path)
        print("[INFO] Loaded RNN emotion model")
    except Exception as e:
        print(f"[WARN] Emotion model not loaded: {e}")

    # ── Sentiment LSTM ────────────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model
        sentiment_model_path = _path("sentiment_rnn_model.h5")
        sentiment_tokenizer_path = _path("sentiment_tokenizer.joblib")
        sentiment_label_path = _path("sentiment_label_encoder.joblib")
        if not (os.path.exists(sentiment_model_path) and os.path.exists(sentiment_tokenizer_path) and os.path.exists(sentiment_label_path)):
            missing = [p for p in [sentiment_model_path, sentiment_tokenizer_path, sentiment_label_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Sentiment model assets missing: {', '.join(missing)}")
        sentiment_model     = load_model(sentiment_model_path, compile=False)
        sentiment_tokenizer = joblib.load(sentiment_tokenizer_path)
        sentiment_le        = joblib.load(sentiment_label_path)
        print("[INFO] Loaded RNN sentiment model")
    except Exception as e:
        print(f"[WARN] Sentiment model not loaded: {e}")

    # ── Intent classifier ─────────────────────────────────────────────────────
    try:
        intent_vectorizer = joblib.load(_path("intent_vectorizer.joblib"))
        intent_clf        = joblib.load(_path("intent_classifier.joblib"))
        intent_le         = joblib.load(_path("intent_label_encoder.joblib"))
        print("[INFO] Loaded intent classifier")
    except Exception as e:
        print(f"[WARN] Intent model not loaded: {e}")


def _seq_pad(tokenizer, text: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")


def predict_emotion(processed_text: str) -> str:
    """Run LSTM emotion model or return 'neutral' as fallback."""
    if emotion_model and emotion_tokenizer and emotion_le:
        try:
            X = _seq_pad(emotion_tokenizer, processed_text)
            probs = emotion_model.predict(X, verbose=0)[0]
            max_prob = np.max(probs)
            if max_prob < 0.65:
                return "neutral"
            return emotion_le.inverse_transform([np.argmax(probs)])[0]
        except Exception as e:
            print(f"[WARN] Emotion predict error: {e}")
    return "neutral"


def predict_sentiment(processed_text: str, emotion: str) -> str:
    """Run LSTM sentiment model; fallback to emotion→sentiment mapping."""
    if sentiment_model and sentiment_tokenizer and sentiment_le:
        try:
            X = _seq_pad(sentiment_tokenizer, processed_text)
            probs = sentiment_model.predict(X, verbose=0)[0]
            return sentiment_le.inverse_transform([np.argmax(probs)])[0]
        except Exception as e:
            print(f"[WARN] Sentiment predict error: {e}")
    # Fallback
    mapping = {
        "joy": "positive", "surprise": "positive",
        "sadness": "negative", "anger": "negative", "fear": "negative",
        "neutral": "neutral",
    }
    return mapping.get(emotion, "neutral")


def predict_intent(processed_text: str, raw_message: str) -> str:
    """TF-IDF + Logistic Regression intent prediction with Context Awareness."""
    
    # 1. Context Override Rules
    study_keywords = ["quiz", "exam", "study", "assignment", "tomorrow"]
    if any(word in raw_message.lower() for word in study_keywords):
        return "studying"
        
    if intent_vectorizer and intent_clf and intent_le:
        try:
            X = intent_vectorizer.transform([processed_text])
            probs = intent_clf.predict_proba(X)[0]
            max_prob = np.max(probs)
            pred_idx = np.argmax(probs)
            predicted_intent = intent_le.inverse_transform([pred_idx])[0]
            
            # 2. Intent Persistence
            if max_prob < 0.65 and SESSION_STATE["active_topic"] == "studying":
                return "studying"
                
            return predicted_intent
        except Exception as e:
            print(f"[WARN] Intent predict error: {e}")
            
    # 3. Fallback to previous intent if available
    if SESSION_STATE["active_topic"] == "studying":
        return "studying"
        
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    raw_message = data.get("message", "").strip()
    if not raw_message:
        return jsonify({"response": "Say something! I'm all ears 👀"}), 200

    # ── NLP Pipeline ──────────────────────────────────────────────────────────
    from preprocess import run_pipeline
    pipeline_out = run_pipeline(raw_message)

    processed   = pipeline_out["processed_text"]
    emoji_emos  = pipeline_out["emoji_emotions"]   # [(emoji, emotion), ...]

    emotion    = predict_emotion(processed)
    sentiment  = predict_sentiment(processed, emotion)
    intent     = predict_intent(processed, raw_message)

    # ── State Management ──────────────────────────────────────────────────────
    # Append to history
    SESSION_STATE["history"].append(raw_message)
    if len(SESSION_STATE["history"]) > 5:
        SESSION_STATE["history"].pop(0)
        
    SESSION_STATE["previous_intent"] = intent
    
    # Topic continuity
    if intent == "studying":
        SESSION_STATE["active_topic"] = "studying"
        SESSION_STATE["topic_turns"] = 3
    else:
        if SESSION_STATE["topic_turns"] > 0:
            SESSION_STATE["topic_turns"] -= 1
        if SESSION_STATE["topic_turns"] == 0:
            SESSION_STATE["active_topic"] = None

    # Emoji emotion override (if strong signal present)
    if emoji_emos:
        emoji_emotion_str = emoji_emos[0][1]  # first detected emoji emotion
        if emoji_emotion_str in ("sadness", "anger", "fear"):
            emotion = emoji_emotion_str
        elif emoji_emotion_str in ("joy", "excitement", "positivity"):
            if emotion == "neutral":
                emotion = "joy"

    # ── Generate response ─────────────────────────────────────────────────────
    from response_generator import generate_response
    response_text = generate_response(
        message        = raw_message,
        emotion        = emotion,
        sentiment      = sentiment,
        intent         = intent,
        history        = SESSION_STATE["history"],
        tokens         = pipeline_out.get("tokens", []),
        emoji_emotions = emoji_emos,
    )

    return jsonify({
        "response":  response_text,
        "emotion":   emotion,
        "sentiment": sentiment,
        "intent":    intent,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def check_groq_connection():
    print("Testing Groq API connection...")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[FAILED] Groq API is NOT connected. Reason: No API key found in .env file.")
        return
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            messages=[{"role": "user", "content": "ping"}],
            model="llama-3.3-70b-versatile",
            max_tokens=5,
        )
        print("[SUCCESS] Groq API connected successfully! Dynamic responses are active.")
    except Exception as e:
        print(f"[FAILED] Groq API is NOT connected. Reason: {e}")

if __name__ == "__main__":
    load_models()
    check_groq_connection()
    app.run(debug=True, port=5000)
