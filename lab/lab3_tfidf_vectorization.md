# Lab 3: TF-IDF Vectorization

## Objective

Improve upon the Bag-of-Words model by implementing TF-IDF (Term Frequency-Inverse Document Frequency). TF-IDF down-weights terms that are common across all documents, leading to more meaningful representations.

## Theory

TF-IDF is the product of two statistics:

1.  **Term Frequency (TF):** Measures how frequently a term appears in a document. This is the raw count, same as in `CountVectorizer`.
    *   `tf(t, d)` = number of times term `t` appears in document `d`.

2.  **Inverse Document Frequency (IDF):** Measures how important a term is. It's calculated by taking the logarithm of the number of documents in the corpus divided by the number of documents where the specific term appears.
    *   `idf(t) = log(N / (dft + 1))` where `N` is the total number of documents and `dft` is the number of documents containing term `t`. We add 1 to the denominator to prevent division-by-zero errors (a technique called smoothing).

3.  **TF-IDF Score:**
    *   `tfidf(t, d) = tf(t, d) * idf(t)`

Finally, it is standard practice to normalize the resulting TF-IDF vectors using L2 normalization.

## Task: TfidfVectorizer Implementation

1.  **Create the file:** `src/representations/tfidf_vectorizer.py`.

2.  **Implement the `TfidfVectorizer` class:**
    *   It should inherit from the `Vectorizer` interface.
    *   The constructor `__init__(self, tokenizer: Tokenizer)` should accept a `Tokenizer`.
    *   It will need attributes for `vocabulary_` and the calculated `idf_` values.

3.  **Implement the `fit` method:**
    *   First, build the vocabulary just like you did in `CountVectorizer`.
    *   Then, calculate the document frequency (DF) for each term in the vocabulary.
    *   Use the DF values to calculate the IDF for each term. Store these in an `idf_` dictionary or list.

4.  **Implement the `transform` method:**
    *   For each document, first compute its term frequency (count) vector, just as in `CountVectorizer`.
    *   Multiply each element (term count) in the vector by its corresponding IDF value from `idf_`.
    *   (Bonus) Apply L2 normalization to the resulting TF-IDF vector.
    *   Return the list of final vectors.

## Evaluation

*   Create a new test file: `test/lab3_test.py`.
*   Use the same corpus from Lab 2.
*   Instantiate your `RegexTokenizer` and `TfidfVectorizer`.
*   Use `fit_transform` on the corpus.
*   Print the learned vocabulary, the IDF values, and the final TF-IDF matrix.
