# Lab 6: Building an NLP Pipeline

## Objective

To integrate the individual NLP components (Tokenizer, Vectorizer, Classifier) developed in previous labs into a single, streamlined pipeline for end-to-end text processing and classification.

## Theory

An **NLP Pipeline** is a sequence of processing steps applied to raw text to achieve a specific NLP task. It chains together various components, where the output of one component serves as the input for the next.

**Benefits of Pipelines:**
*   **Modularity:** Each component is independent and can be swapped out.
*   **Reusability:** The entire pipeline or its parts can be reused across different projects.
*   **Consistency:** Ensures the same sequence of operations is applied every time.
*   **Simplicity:** Simplifies the code for complex workflows.

## Task 1: TextPipeline Implementation

1.  **Create the file:** `src/pipelines/text_pipeline.py`.

2.  **Implement the `TextPipeline` class:**
    *   The constructor `__init__(self, tokenizer: Tokenizer, vectorizer: Vectorizer, classifier: TextClassifier)` should accept instances of your `Tokenizer`, `Vectorizer`, and `TextClassifier` classes.

3.  **Implement `process(self, text: str) -> int`:**
    *   This method will take a single raw text string.
    *   It should use the `tokenizer` to tokenize the text.
    *   Then, it should use the `vectorizer` to transform the tokenized text into a numerical feature vector. (Remember `vectorizer.transform` expects a list of strings, so you'll need to pass `[text]` and handle the output).
    *   Finally, it should use the `classifier` to predict the label from the feature vector. (Similarly, `classifier.predict` expects a list of feature vectors).
    *   Return the predicted label (integer).

4.  **Implement `process_batch(self, texts: List[str]) -> List[int]`:**
    *   This method will take a list of raw text strings.
    *   It should tokenize all texts.
    *   Vectorize all texts.
    *   Classify all texts.
    *   Return a list of predicted labels.

## Evaluation

*   Create a new test file: `test/lab6_test.py`.
*   Define the same sample `texts` and `labels` dataset from Lab 5.
*   Instantiate your `RegexTokenizer`, `TfidfVectorizer`, and `TextClassifier`.
*   Train the `TextClassifier` using the training data (as done in Lab 5).
*   Instantiate your `TextPipeline` with these trained components.
*   Test the `process` method with a single new sentence.
*   Test the `process_batch` method with the test set from Lab 5 and print the predictions.
