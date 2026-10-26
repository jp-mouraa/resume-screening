# Resume Classification with NLP & Machine Learning

A machine learning project that classifies resumes into job categories using NLP techniques and traditional ML models.

## Overview

This project performs exploratory data analysis (EDA) and builds classification models to categorize resumes into professional areas (e.g., Data Science, Java Developer, DevOps Engineer, HR, etc.) based on their text content.

The dataset contains **962 resumes** across **25 job categories**.

## Dataset

The dataset (`resume_dataset.csv`) contains two columns:
- `Category` — job area label
- `Resume` — raw resume text

## Pipeline

```
Raw Resumes → Text Preprocessing (NLTK) → Vectorization (CountVectorizer) → Classification (MNB / KNN)
```

## How to Run

```bash
conda create -n resume-screening python=3.11
conda activate resume-screening
pip install pandas numpy nltk scikit-learn seaborn matplotlib plotly nbformat ipykernel
```

Then open and run `solution.ipynb` in Jupyter or Google Colab.

## Features

- **Exploratory Data Analysis** — null value checks, category distribution, statistical summaries
- **Text Preprocessing** — tokenization and stopword removal using NLTK
- **Feature Engineering** — text vectorization with `CountVectorizer`
- **Classification** — two models trained and evaluated:
  - Multinomial Naive Bayes (MNB)
  - K-Nearest Neighbors (KNN)
- **Binary Classification** — resumes also classified as `TECH` vs `OTHERS`

## Results

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes | 98% |
| K-Nearest Neighbors (k=3) | 100% |

## Notes

The final section of the notebook discusses a potential extension using **vector databases and LLMs** to generate synthetic resumes for classification.
