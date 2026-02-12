# Fake News Detection — Fast & Lightweight Linear Model

## 📌 Overview
This project provides a highly efficient machine learning solution for fake news classification. It is optimized for medium-to-large datasets (up to ~200k rows) where accuracy and interpretability are paramount.

Unlike complex hybrid models or heavy deep learning transformers, this approach uses a streamlined **Linear Support Vector Machine (SVM)** pipeline. It achieves state-of-the-art performance on text classification tasks while remaining computationally inexpensive.

## 🚀 Key Features
* **High-Precision Vectorization:** Uses `TfidfVectorizer` with an expanded feature space (50,000 features) to capture subtle semantic cues.
* **Optimized for Stability:** The `LinearSVC` model provides more consistent convergence than stochastic methods on this data scale.
* **Interpretable Features:** Maintains a direct mapping between words and importance scores, unlike hashing-based methods.
* **Resource Efficient:** Runs comfortably on standard laptops or Google Colab free tiers without crashing RAM.

## 🛠️ Dependencies
* Python 3.x
* pandas
* scikit-learn

## ⚙️ How to Run
1.  Ensure `train.csv` and `test.csv` are in the same directory.
2.  Run the script:
    ```bash
    python fake_news_linear.py
    ```
3.  The script will generate `submission.csv` containing the predictions.

## 📊 Pipeline Architecture

### 1. Preprocessing
* **Text Normalization:** Standardizes text case (lowercase) and removes excessive whitespace.
* **Concatenation:** Merges `Title` and `Text` columns to ensure the model catches contradictions between headlines and article bodies.
* **Label Encoding:** Converts various label formats ("fake", "REAL", "0", "1") into a unified binary integer format.

### 2. Feature Engineering
* **Algorithm:** Term Frequency-Inverse Document Frequency (TF-IDF).
* **Configuration:**
    * `ngram_range=(1,2)`: Captures single words ("election") and pairs ("election fraud") to understand context.
    * `max_features=50,000`: Tracks the top 50k most significant terms, filtering out rare noise.
    * `sublinear_tf=True`: Applies logarithmic scaling to term frequency (1 + log(tf)), preventing long articles from skewing results.

### 3. Classification Model
* **Algorithm:** Linear Support Vector Classification (LinearSVC).
* **Regularization:** L2 penalty (default) with `C=1.0`.
* **Solver:** Liblinear (optimized for linear classification on large sparse datasets).
