# Lab 5: Text Classification

## Objective

To build a complete text classification pipeline, from raw text to a trained machine learning model, using the tokenization and vectorization techniques learned in previous labs.

## Theory

**Text Classification** is the task of assigning categories or labels to text documents. Common applications include sentiment analysis, spam detection, and topic labeling.

We will use a **supervised learning** approach, meaning we will train our model on a dataset where each text document is already associated with a known label.

**Pipeline:**
Raw Text -> Tokenization -> Vectorization -> Machine Learning Model -> Prediction

**Model:** We will use **Logistic Regression**, a simple yet effective linear model for binary classification.

**Evaluation:** To assess our model's performance, we will use metrics such as Accuracy, Precision, Recall, and F1-score.

## Task 1: Data Preparation

1.  **Dataset:** We'll use a small, in-memory dataset for simplicity.
    ```python
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative
    ```
2.  **Vectorize:** Use your `TfidfVectorizer` (or `CountVectorizer`) from Lab 3 (or Lab 2) to transform these texts into numerical features.

## Task 2: TextClassifier Implementation

1.  **Create the file:** `src/models/text_classifier.py`.

2.  **Implement the `TextClassifier` class:**
    *   The constructor `__init__(self, vectorizer: Vectorizer)` should accept a `Vectorizer` instance.
    *   It should have an attribute `_model` to store the trained `LogisticRegression` model from `scikit-learn`.

3.  **Implement `fit(self, texts: List[str], labels: List[int])`:**
    *   This method will train the classifier.
    *   First, use the `vectorizer` to `fit_transform` the input `texts` into a feature matrix `X`.
    *   Initialize a `LogisticRegression` model (e.g., `solver='liblinear'` for small datasets).
    *   Train the model using `model.fit(X, labels)`.

4.  **Implement `predict(self, texts: List[str]) -> List[int]`:**
    *   This method will make predictions on new texts.
    *   First, use the `vectorizer` to `transform` the input `texts` into a feature matrix `X`.
    *   Use the trained `_model` to predict labels: `_model.predict(X)`.

5.  **Implement `evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]`:**
    *   This method will calculate evaluation metrics.
    *   Use `sklearn.metrics` functions (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`) to compute and return a dictionary of metrics.

## Evaluation

*   Create a new test file: `test/lab5_test.py`.
*   Define the `texts` and `labels` dataset.
*   Split the data into training and testing sets (e.g., 80% train, 20% test). You can use `sklearn.model_selection.train_test_split`.
*   Instantiate your `RegexTokenizer` and `TfidfVectorizer`.
*   Instantiate your `TextClassifier` with the vectorizer.
*   Train the classifier using the training data.
*   Make predictions on the test data.
*   Evaluate the predictions and print the metrics.
