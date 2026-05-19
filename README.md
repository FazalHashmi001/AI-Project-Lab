# Resume Parser AI

A desktop AI application that parses resumes, predicts the job category, and assigns a score from 0 to 10 using machine learning.

## Tech Stack

| Component     | Technology              |
|---------------|-------------------------|
| Language      | Python 3.x              |
| ML Model      | K-Nearest Neighbors (KNN) |
| Vectorization | TF-IDF (scikit-learn)   |
| Preprocessing | NLTK                    |
| GUI           | Tkinter                 |
| Visualization | Matplotlib              |

## Dataset

[Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

- Columns: `Category`, `Resume`
- Download the CSV and select it when prompted inside the app.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/resume-parser-ai.git
cd resume-parser-ai
```

### 2. Create a virtual environment (recommended)

```bash
conda create -n resume_env python=3.10
conda activate resume_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python main.py
```

## How to Use

1. Click **Browse & Load Dataset** → select the Kaggle CSV file.
2. Wait for training to complete (status bar shows accuracy).
3. Paste any resume text in the text box.
4. Click **Analyse Resume**.
5. View the predicted category, score (0–10), and charts.

## Project Structure

```
resume_parser/
├── main.py          # Entry point
├── gui.py           # Tkinter GUI
├── model.py         # KNN model + TF-IDF
├── preprocessor.py  # NLTK preprocessing pipeline
├── visualizer.py    # Matplotlib charts
├── requirements.txt
└── README.md
```

## ML Pipeline

```
Raw Resume Text
     ↓
NLTK Preprocessing (lowercase → clean → tokenize → stopword removal → lemmatize)
     ↓
TF-IDF Vectorization (top 1500 features)
     ↓
KNN Classifier (k=5)
     ↓
Category Prediction + Confidence Score (0–10)
```
