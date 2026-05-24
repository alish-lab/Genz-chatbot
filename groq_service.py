"""
groq_service.py — Groq API client for GenZBot
─────────────────────────────────────────────
Handles: retry with exponential backoff, error handling.

NOTE: Debounce removed — it was blocking Pass 2 of the two-pass pipeline.
Since Pass 1 and Pass 2 fire sequentially within the same request, the
debounce delay was causing Pass 2 to return None, leaving responses in
raw unformatted English. Rate limiting is handled by retry logic instead.

NOTE: LRU cache also removed — was returning stale responses when the same
user message appeared with different NLP-derived system prompts.
"""

import os
import time
import logging
from groq import Groq, APIConnectionError, RateLimitError, APIError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GroqService")

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if api_key:
            _client = Groq(api_key=api_key)
    return _client


def generate_ai_response(system_prompt: str, user_message: str) -> str:
    """
    Generate an AI response using Groq.
    Handles retries with exponential backoff.

    Returns the response string, or None on failure.
    """

    client = get_client()
    if not client:
        logger.warning("GROQ_API_KEY not set. Using fallback responses.")
        return None

    max_retries = 3
    base_delay  = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=200,       # keep responses concise
                temperature=0.7,      # some creativity but not too wild
            )
            content = response.choices[0].message.content
            if content:
                content = content.strip()
            logger.info("Groq API response received successfully.")
            return content or None

        except RateLimitError as e:
            logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached for RateLimitError.")
                return None

        except APIConnectionError as e:
            logger.warning(f"Connection error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached for APIConnectionError.")
                return None

        except APIError as e:
            logger.error(f"Groq API error: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error calling Groq API: {e}")
            return None

    return None
