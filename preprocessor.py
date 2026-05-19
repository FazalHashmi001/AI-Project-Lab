"""
preprocessor.py
Handles all text cleaning and feature extraction from resume text.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class ResumePreprocessor:
    """
    Cleans and prepares resume text for model input.
    Steps: lowercase -> remove noise -> tokenize -> remove stopwords -> lemmatize
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Remove URLs, emails, special characters, and extra spaces."""
        text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        # Keep letters and spaces only
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_filter(self, text):
        """Tokenize, remove stopwords, and lemmatize."""
        tokens = word_tokenize(text)
        filtered = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return filtered

    def preprocess(self, text):
        """Full pipeline: clean text and return processed token list."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return tokens

    def preprocess_to_string(self, text):
        """Return processed tokens joined as a single string (for TF-IDF)."""
        tokens = self.preprocess(text)
        result = ' '.join(tokens)

        # Fallback 1: preprocessing removed everything, use basic cleaned text
        if not result.strip():
            result = self.clean_text(str(text))

        # Fallback 2: still empty, return placeholder
        if not result.strip():
            result = 'unknown resume'

        return result