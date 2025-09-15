# Lab 13: Advanced Text Preprocessing (Stemming, Lemmatization, and POS Tagging)

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab requires specific data for NLTK and spaCy models. You can download them by running the following commands in your Python environment:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

## Objective

To deepen your understanding of text preprocessing by implementing stemming and lemmatization, and performing Part-of-Speech (POS) tagging. These techniques are crucial for normalizing text and extracting linguistic features.

## Theory

**Text Normalization:** The process of converting text into a canonical (standard) form. This helps reduce the vocabulary size and allows the model to treat different forms of a word as the same.

1.  **Stemming:**
    *   The process of reducing inflected (or sometimes derived) words to their word stem, base or root form—generally a written word form.
    *   Often involves simply chopping off suffixes.
    *   **Example:** "running", "runs", "ran" -> "run"
    *   **Algorithm:** Porter Stemmer is a common algorithm.

2.  **Lemmatization:**
    *   The process of grouping together the inflected forms of a word so they can be analyzed as a single item, identified by the word's lemma, or dictionary form.
    *   More sophisticated than stemming, as it uses a vocabulary and morphological analysis.
    *   **Example:** "better" -> "good", "ran" -> "run"

**Part-of-Speech (POS) Tagging:**
*   The process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context.
*   **Examples:** Noun (NN), Verb (VB), Adjective (JJ), Adverb (RB).
*   **Importance:** Useful for syntactic analysis, named entity recognition, and disambiguation.

## Task 1: Stemming with NLTK

1.  **Create the file:** `src/preprocessing/advanced_preprocessor.py`.

2.  **Implement the `AdvancedPreprocessor` class:**
    *   The constructor `__init__(self, tokenizer: Tokenizer)` should accept a `Tokenizer` instance.
    *   Initialize an `NLTK` Porter Stemmer (`nltk.stem.PorterStemmer`).

3.  **Implement `stem_text(self, text: str) -> List[str]`:**
    *   Tokenize the input `text` using the provided `tokenizer`.
    *   Apply the Porter Stemmer to each token.
    *   Return a list of stemmed tokens.

## Task 2: Lemmatization with spaCy

1.  **Implement `lemmatize_text(self, text: str) -> List[str]`:**
    *   Load a `spaCy` language model (e.g., `en_core_web_sm`). You can reuse the loading logic from Lab 7.
    *   Process the text with `spaCy`.
    *   Extract the lemma for each token.
    *   Return a list of lemmas.

## Task 3: POS Tagging with spaCy

1.  **Implement `pos_tag_text(self, text: str) -> List[Dict[str, str]]`:**
    *   Process the text with the same `spaCy` model used for lemmatization.
    *   For each token, extract its text and its Part-of-Speech tag.
    *   Return a list of dictionaries (e.g., `{"token": "running", "pos": "VERB"}`).

## Evaluation

*   Create a new test file: `test/lab13_test.py`.
*   Instantiate your `RegexTokenizer` and `AdvancedPreprocessor`.
*   Test each implemented method with the following sentence and print the results:
    *   `"The quick brown foxes are running quickly to their dens." `
