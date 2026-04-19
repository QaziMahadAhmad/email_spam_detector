"""
utils/text_processing.py
Reusable text-cleaning helpers shared by trainer and predictor.
"""

import re
import string

# Common English stop words (no external dependency required)
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "are",
    "was", "be", "as", "by", "from", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "i", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "our", "their", "so", "if",
    "then", "than", "up", "out", "about", "into", "its", "there",
    "what", "which", "who", "when", "how", "all", "been",
}


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation and digits
    4. Strip extra whitespace
    5. Remove stop words
    6. Simple suffix-stripping stemmer
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [_stem(t) for t in tokens]
    return " ".join(tokens)


def _stem(word: str) -> str:
    """
    Very simple suffix-stripping stemmer.
    Handles the most common English suffixes without NLTK.
    """
    for suffix in ("ing", "tion", "ness", "ment", "ful", "less",
                   "able", "ible", "ous", "ive", "ers", "es", "ed", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word
