"""
model.py
Trains a KNN classifier on the resume dataset and provides
a scoring function (0-10) for new resumes.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessor import ResumePreprocessor


class ResumeModel:
    """
    Encapsulates the full ML pipeline:
    TF-IDF vectorizer + KNN classifier + confidence-based scoring.
    """

    def __init__(self, n_neighbors=5):
        self.preprocessor = ResumePreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=1500, min_df=1)
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.classes_ = []

    def _build_resume_text(self, row):
        """Combine relevant columns into one resume text string."""
        parts = []
        for col in ['career_objective', 'skills', 'responsibilities',
                    'positions', 'major_field_of_studies', 'degree_names']:
            val = str(row.get(col, ''))
            if val and val.lower() not in ('nan', 'none', ''):
                parts.append(val)
        return ' '.join(parts)

    def _get_category_col(self, df):
        """Find the job position/category column."""
        for col in df.columns:
            if 'job_position' in col.lower() or col.strip() == 'Category':
                return col
        # fallback: first column
        return df.columns[0]

    def load_and_train(self, csv_path):
        """
        Load the resume dataset CSV and train the model.
        Works with both the Kaggle 2-column format and
        the multi-column format (career_objective, skills, etc.)
        Returns: (X_test, y_test) for evaluation.
        """
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        df.dropna(subset=[self._get_category_col(df)], inplace=True)
        df.reset_index(drop=True, inplace=True)

        category_col = self._get_category_col(df)

        # Build resume text depending on dataset format
        if 'Resume' in df.columns:
            # Kaggle 2-column format
            df['resume_text'] = df['Resume'].astype(str)
        else:
            # Multi-column format — combine relevant fields
            df['resume_text'] = df.apply(self._build_resume_text, axis=1)

        # Preprocess text
        df['processed'] = df['resume_text'].apply(
            self.preprocessor.preprocess_to_string)

        # Drop rows where processed text is still empty
        df = df[df['processed'].str.strip().str.len() > 0]
        df.reset_index(drop=True, inplace=True)

        # Encode labels
        df['label'] = self.label_encoder.fit_transform(df[category_col])
        self.classes_ = self.label_encoder.classes_

        # Vectorize
        X = self.vectorizer.fit_transform(df['processed']).toarray()
        y = df['label'].values

        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        return X_test, y_test

    def evaluate(self, X_test, y_test):
        """Return accuracy, precision, recall, F1 on the test set."""
        y_pred = self.classifier.predict(X_test)
        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1':        f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        return metrics

    def predict_category(self, resume_text):
        """Predict job category for a given resume string."""
        processed = self.preprocessor.preprocess_to_string(resume_text)
        vector = self.vectorizer.transform([processed]).toarray()
        label_idx = self.classifier.predict(vector)[0]
        category = self.label_encoder.inverse_transform([label_idx])[0]
        return category

    def score_resume(self, resume_text):
        """
        Assign a score from 0 to 10 based on KNN prediction confidence.
        Confidence = fraction of k nearest neighbors that agree on the class.
        Score = confidence * 10, rounded to 1 decimal.
        """
        processed = self.preprocessor.preprocess_to_string(resume_text)
        vector = self.vectorizer.transform([processed]).toarray()

        proba = self.classifier.predict_proba(vector)[0]
        confidence = float(np.max(proba))

        score = round(confidence * 10, 1)
        score = max(0.0, min(10.0, score))
        return score