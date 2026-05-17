"""
train.py — Model training for GenZBot
────────────────────────────────────────────────────────────────
Models trained:
  1. Emotion detection  → LSTM(64) on tweet emotion dataset
                          labels: joy, sadness, anger, fear, surprise, neutral
  2. Sentiment analysis → LSTM(64) on tweet emotion dataset
                          labels: positive, negative, neutral
  3. Intent classifier  → TF-IDF + Logistic Regression (rule-seeded data)

All models saved to ./models/
────────────────────────────────────────────────────────────────
"""

import os
import re
import joblib
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF noise

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Dataset paths ────────────────────────────────────────────────────────────
# The loader checks multiple candidate locations so the project works whether
# you run train.py from inside genzbot/ or from the parent folder.
def _find_emotion_file(filename: str) -> str:
    """Search common locations for the emotion dataset file."""
    candidates = [
        # Relative to this file (genzbot/data/emotion_de/)
        os.path.join(BASE_DIR, "data", "emotion_de", filename),
        # One level up (parent/emotion de/)
        os.path.join(BASE_DIR, "..", "emotion de", filename),
        os.path.join(BASE_DIR, "..", "emotion_de", filename),
        # Same folder as train.py
        os.path.join(BASE_DIR, filename),
        # Current working directory
        os.path.join(os.getcwd(), "emotion_de", filename),
        os.path.join(os.getcwd(), "emotion de", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Return default path (will show a clear warning if missing)
    return os.path.join(BASE_DIR, "data", "emotion_de", filename)

EMOTION_TRAIN = _find_emotion_file("train.txt")
EMOTION_VAL   = _find_emotion_file("val.txt")
EMOTION_TEST  = _find_emotion_file("test.txt")

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MAX_WORDS   = 10_000
MAX_LEN     = 50
EMBED_DIM   = 64
LSTM_UNITS  = 64
BATCH_SIZE  = 32
EPOCHS      = 15

# ── Emotion → sentiment mapping ───────────────────────────────────────────────
EMOTION_TO_SENTIMENT = {
    "joy":      "positive",
    "surprise": "positive",
    "sadness":  "negative",
    "anger":    "negative",
    "fear":     "negative",
    "neutral":  "neutral",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_emotion_file(path: str):
    """
    Each line: <text>;<label>
    Returns (texts, labels).
    """
    texts, labels = [], []
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return texts, labels
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            parts = line.rsplit(";", 1)
            if len(parts) == 2:
                text, label = parts
                texts.append(_simple_clean(text))
                labels.append(label.strip().lower())
    return texts, labels


def _load_all_emotion_data():
    train_x, train_y = _load_emotion_file(EMOTION_TRAIN)
    val_x,   val_y   = _load_emotion_file(EMOTION_VAL)
    test_x,  test_y  = _load_emotion_file(EMOTION_TEST)
    return train_x, train_y, val_x, val_y, test_x, test_y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Emotion LSTM
# ─────────────────────────────────────────────────────────────────────────────

def train_emotion_lstm():
    print("\n" + "="*60)
    print("  TRAINING: Emotion Detection LSTM")
    print("="*60)

    train_x, train_y, val_x, val_y, test_x, test_y = _load_all_emotion_data()

    if not train_x:
        print("[ERROR] No emotion training data found. Skipping.")
        return

    # Label encoding
    le = LabelEncoder()
    le.fit(train_y + val_y + test_y)
    num_classes = len(le.classes_)
    print(f"  Emotion classes ({num_classes}): {list(le.classes_)}")

    y_train_enc = to_categorical(le.transform(train_y), num_classes)
    y_val_enc   = to_categorical(le.transform(val_y),   num_classes)
    y_test_enc  = to_categorical(le.transform(test_y),  num_classes)

    # Tokenization
    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(train_x)

    X_train = pad_sequences(tok.texts_to_sequences(train_x), maxlen=MAX_LEN, padding="post", truncating="post")
    X_val   = pad_sequences(tok.texts_to_sequences(val_x),   maxlen=MAX_LEN, padding="post", truncating="post")
    X_test  = pad_sequences(tok.texts_to_sequences(test_x),  maxlen=MAX_LEN, padding="post", truncating="post")

    # Model: Embedding → LSTM(64) → Dense
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="emotion_lstm")

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    model.fit(
        X_train, y_train_enc,
        validation_data=(X_val, y_val_enc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1,
    )

    # Evaluation on test set
    print("\n── Test Evaluation ──")
    y_pred_prob = model.predict(X_test)
    y_pred = le.inverse_transform(np.argmax(y_pred_prob, axis=1))
    y_true = test_y

    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy : {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=le.classes_))

    # Save
    model.save(os.path.join(MODEL_DIR, "emotion_rnn_model.h5"))
    joblib.dump(tok, os.path.join(MODEL_DIR, "emotion_tokenizer.joblib"))
    joblib.dump(le,  os.path.join(MODEL_DIR, "emotion_label_encoder.joblib"))
    print("\n  ✅ Emotion model saved to models/")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sentiment LSTM  (derived from emotion dataset via label mapping)
# ─────────────────────────────────────────────────────────────────────────────

def train_sentiment_lstm():
    print("\n" + "="*60)
    print("  TRAINING: Sentiment Analysis LSTM")
    print("="*60)

    train_x, train_y_emo, val_x, val_y_emo, test_x, test_y_emo = _load_all_emotion_data()

    if not train_x:
        print("[ERROR] No emotion data for sentiment training. Skipping.")
        return

    # Map emotion labels → sentiment labels
    def to_sentiment(labels):
        return [EMOTION_TO_SENTIMENT.get(l, "neutral") for l in labels]

    train_y = to_sentiment(train_y_emo)
    val_y   = to_sentiment(val_y_emo)
    test_y  = to_sentiment(test_y_emo)

    le = LabelEncoder()
    le.fit(["positive", "negative", "neutral"])
    num_classes = len(le.classes_)
    print(f"  Sentiment classes: {list(le.classes_)}")

    y_train_enc = to_categorical(le.transform(train_y), num_classes)
    y_val_enc   = to_categorical(le.transform(val_y),   num_classes)
    y_test_enc  = to_categorical(le.transform(test_y),  num_classes)

    # Class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(train_y),
                              y=train_y)
    class_weights = {i: w for i, w in enumerate(cw)}

    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(train_x)

    X_train = pad_sequences(tok.texts_to_sequences(train_x), maxlen=MAX_LEN, padding="post", truncating="post")
    X_val   = pad_sequences(tok.texts_to_sequences(val_x),   maxlen=MAX_LEN, padding="post", truncating="post")
    X_test  = pad_sequences(tok.texts_to_sequences(test_x),  maxlen=MAX_LEN, padding="post", truncating="post")

    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(LSTM_UNITS)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="sentiment_lstm")

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    model.fit(
        X_train, y_train_enc,
        validation_data=(X_val, y_val_enc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[es],
        verbose=1,
    )

    # Evaluation
    print("\n── Test Evaluation ──")
    y_pred_prob = model.predict(X_test)
    y_pred = le.inverse_transform(np.argmax(y_pred_prob, axis=1))

    acc = accuracy_score(test_y, y_pred)
    print(f"  Accuracy : {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(test_y, y_pred, target_names=le.classes_))
    print("\n  Confusion Matrix:")
    print(confusion_matrix(test_y, y_pred, labels=le.classes_))

    # Save
    model.save(os.path.join(MODEL_DIR, "sentiment_rnn_model.h5"))
    joblib.dump(tok, os.path.join(MODEL_DIR, "sentiment_tokenizer.joblib"))
    joblib.dump(le,  os.path.join(MODEL_DIR, "sentiment_label_encoder.joblib"))
    print("\n  ✅ Sentiment model saved to models/")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Intent Classifier  (TF-IDF + Logistic Regression, rule-seeded)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_DATA = [
    # greeting
    ("hey", "greeting"), ("hello", "greeting"), ("hi there", "greeting"),
    ("what's up", "greeting"), ("sup", "greeting"), ("yo", "greeting"),
    ("good morning", "greeting"), ("good night", "greeting"),
    ("how are you", "greeting"), ("how you doing", "greeting"),
    ("wassup", "greeting"), ("heyyyy", "greeting"),

    # studying
    ("help me study", "studying"), ("i have an exam", "studying"),
    ("explain recursion", "studying"), ("what is machine learning", "studying"),
    ("how does gradient descent work", "studying"),
    ("can you quiz me", "studying"), ("i don't understand derivatives", "studying"),
    ("study session", "studying"), ("help with homework", "studying"),
    ("i need to revise", "studying"), ("calculus is hard", "studying"),
    ("explain oop", "studying"), ("what is a neural network", "studying"),

    # emotional support
    ("i feel sad", "emotional_support"), ("im really down", "emotional_support"),
    ("i'm stressed out", "emotional_support"), ("nobody gets me", "emotional_support"),
    ("i need someone to talk to", "emotional_support"),
    ("i'm overwhelmed", "emotional_support"), ("i feel lost", "emotional_support"),
    ("everything feels too much", "emotional_support"),
    ("i'm really anxious", "emotional_support"),
    ("im going through something", "emotional_support"),
    ("i feel like crying", "emotional_support"),

    # jokes
    ("tell me a joke", "jokes"), ("make me laugh", "jokes"),
    ("do you know any memes", "jokes"), ("something funny", "jokes"),
    ("roast me", "jokes"), ("give me a fun fact", "jokes"),
    ("say something silly", "jokes"), ("i'm bored", "jokes"),
    ("entertain me", "jokes"),

    # programming
    ("debug this code", "programming"), ("write a python function", "programming"),
    ("what is a list comprehension", "programming"),
    ("my code has a bug", "programming"), ("how do i use async await", "programming"),
    ("javascript vs python", "programming"),
    ("explain sql joins", "programming"), ("how do decorators work", "programming"),
    ("what is a rest api", "programming"), ("write a flask route", "programming"),

    # general conversation
    ("what do you think about", "general"), ("just chatting", "general"),
    ("tell me something interesting", "general"), ("your favorite movie", "general"),
    ("do you like music", "general"), ("what is your opinion on", "general"),
    ("let's talk", "general"), ("random question", "general"),
    ("what should i do today", "general"), ("i'm bored help", "general"),
]


def train_intent_classifier():
    print("\n" + "="*60)
    print("  TRAINING: Intent Classifier (TF-IDF + Logistic Regression)")
    print("="*60)

    texts  = [d[0] for d in INTENT_DATA]
    labels = [d[1] for d in INTENT_DATA]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=500, C=5.0)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"  Train accuracy : {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "intent_vectorizer.joblib"))
    joblib.dump(clf,        os.path.join(MODEL_DIR, "intent_classifier.joblib"))
    joblib.dump(le,         os.path.join(MODEL_DIR, "intent_label_encoder.joblib"))
    print("\n  ✅ Intent model saved to models/")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train all
# ─────────────────────────────────────────────────────────────────────────────

def train_all():
    print("\n🚀 GenZBot Model Training Started\n")
    train_emotion_lstm()
    train_sentiment_lstm()
    train_intent_classifier()
    print("\n✅ All models trained and saved to ./models/\n")


if __name__ == "__main__":
    train_all()
