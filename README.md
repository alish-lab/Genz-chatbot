# GenZBot

A Gen Z–style chatbot built with NLP, slang normalization, emoji understanding, emotion detection, sentiment analysis, and intent-aware response generation.

## Features

- Full NLP preprocessing pipeline for slang, acronyms, emoji meaning, and text cleanup
- Emotion detection with LSTM models
- Sentiment analysis with LSTM models
- Intent classification for chat behavior
- Flask web UI with dark theme in `templates/index.html`
- Modular training and inference workflow

## Repository Structure

- `app.py` — Flask application and inference server
- `preprocess.py` — text normalization, slang mapping, emoji parsing, and tokenization
- `train.py` — model training workflow for emotion/sentiment/intent
- `response_generator.py` — Gen Z response creation and adaptation logic
- `groq_service.py` — external API/service integration helper
- `retrieval.py` — retrieval support for answers or contextual data
- `requirements.txt` — Python dependencies
- `data/` — input datasets and resources
- `models/` — trained model artifacts
- `templates/` — web UI templates

## Getting Started

1. Create and activate a Python virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Add a `.env` file in the project root with any required secrets, for example:

```text
GEMINI_API_KEY=your_api_key_here
```

4. Run the application:

```powershell
python app.py
```

5. Open the chatbot at `http://127.0.0.1:5000`

## Training

To retrain models, place your emotion dataset files in `data/emotion_de/` and run:

```powershell
python train.py
```

Expected files:

- `data/emotion_de/train.txt`
- `data/emotion_de/val.txt`
- `data/emotion_de/test.txt`

Each line should follow the format:

```text
message_text;label
```

Supported emotion labels: `joy`, `sadness`, `anger`, `fear`, `surprise`, `neutral`

## Notes

- `.env` and `__pycache__/` are ignored by `.gitignore`
- `models/` contains generated artifacts and may be large
- `data/chat_dataset.csv` is currently untracked and may contain additional training or chat data

UI
<img width="1223" height="832" alt="image" src="https://github.com/user-attachments/assets/6dfd3d7b-67be-4696-8ff6-26f55a43f638" />

