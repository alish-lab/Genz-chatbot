import os
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "genz_slang_dataset_final2020_2026.csv")

# Global caches to avoid repeated loading
_vectorizer = None
_tfidf_matrix = None
_dataset = []

def _init_retrieval_module():
    """
    Loads the dataset and fits the TF-IDF vectorizer only once.
    Combines 'meaning' and 'example_sentence' to form the document corpus.
    """
    global _vectorizer, _tfidf_matrix, _dataset
    
    if _vectorizer is not None:
        return  # Already initialized

    _dataset = []
    corpus = []
    
    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meaning = row.get("meaning", "").strip()
                example = row.get("example_sentence", "").strip()
                
                if meaning and example:
                    # Append to dataset cache for fast retrieval
                    _dataset.append({
                        "meaning": meaning,
                        "example": example
                    })
                    # Combine meaning and example for robust vectorization matching
                    corpus.append(f"{meaning} {example}")
                    
        # Initialize and fit TF-IDF
        _vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        _tfidf_matrix = _vectorizer.fit_transform(corpus)
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize TF-IDF retrieval module: {e}")

def get_top_genz_examples(normal_response: str, top_n: int = 5, confidence_threshold: float = 0.05) -> str:
    """
    Takes a normal LLM response, vectorizes it, and retrieves the top N 
    semantically similar Gen-Z examples from the dataset using cosine similarity.
    
    Returns a formatted string of examples to inject into the rewrite prompt.
    Falls back gracefully if confidence is too low.
    """
    _init_retrieval_module()
    
    if not _vectorizer or not _dataset or not normal_response:
        return "No context examples available."
        
    try:
        # Vectorize the input response
        query_vec = _vectorizer.transform([normal_response])
        
        # Compute cosine similarity between the query and the entire dataset
        similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
        
        # Get indices of the top N highest similarities
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        examples_found = []
        for rank, idx in enumerate(top_indices):
            score = similarities[idx]
            
            # If the highest score is below the threshold, skip adding to avoid irrelevant spam
            if score < confidence_threshold:
                continue
                
            item = _dataset[idx]
            formatted_example = (
                f"Example {len(examples_found) + 1}:\n"
                f"Meaning: {item['meaning']}\n"
                f"Example sentence: {item['example']}\n"
            )
            examples_found.append(formatted_example)
            
        if not examples_found:
            return "No highly relevant context examples found."
            
        # Combine the formatted examples
        return "\n".join(examples_found)
        
    except Exception as e:
        print(f"[ERROR] TF-IDF Retrieval failed: {e}")
        return "No context examples available."
