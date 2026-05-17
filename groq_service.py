import os
import time
import logging
from functools import lru_cache
from groq import Groq, APIConnectionError, RateLimitError, APIError

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GroqService")

# Initialize client lazily to avoid crashing on import if no key
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if api_key:
            _client = Groq(api_key=api_key)
    return _client

# Debounce tracker
_LAST_REQUEST_TIME = 0.0
_DEBOUNCE_DELAY_SECONDS = 0.5  # Prevents rapid-fire duplicate requests

@lru_cache(maxsize=100)
def _generate_content_cached(system_prompt: str, user_message: str) -> str:
    """Cached internal method to save API calls for identical prompts."""
    client = get_client()
    if not client:
        logger.warning("GROQ_API_KEY not set. Using fallback responses.")
        return None

    # Retry logic configuration
    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.3-70b-versatile",
            )
            logger.info("Successfully generated dynamic response from Groq API.")
            return response.choices[0].message.content.strip() or None

        except RateLimitError as e:
            logger.warning(f"Groq API rate limit exceeded (Attempt {attempt+1}/{max_retries}). Error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached for RateLimitError. Failing gracefully.")
                return None
                
        except APIConnectionError as e:
            logger.warning(f"Groq API connection error (Attempt {attempt+1}/{max_retries}). Error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached for APIConnectionError. Failing gracefully.")
                return None
                
        except APIError as e:
            logger.error(f"Groq API error encountered: {e}. Failing gracefully.")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error calling Groq API: {e}")
            return None

def generate_ai_response(system_prompt: str, user_message: str) -> str:
    """
    Public entry point for generating AI responses using Groq.
    Handles debouncing, caching, and retries.
    """
    global _LAST_REQUEST_TIME
    
    current_time = time.time()
    if current_time - _LAST_REQUEST_TIME < _DEBOUNCE_DELAY_SECONDS:
        logger.warning("Request debounced. Too many requests in rapid succession.")
        return None
        
    _LAST_REQUEST_TIME = current_time
    
    # Calls the cached internal function
    return _generate_content_cached(system_prompt, user_message)
