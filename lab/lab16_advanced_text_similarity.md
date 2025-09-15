# Lab 16: Advanced Text Similarity

## Objective

To explore different methods for calculating text similarity beyond simple cosine similarity of TF-IDF vectors, including Jaccard similarity and semantic similarity using word embeddings.

## Theory

**Text Similarity** is a fundamental concept in NLP, used in tasks like information retrieval, plagiarism detection, and document clustering. While cosine similarity on TF-IDF vectors is a common baseline, it has limitations, especially in capturing semantic meaning.

1.  **Jaccard Similarity:**
    *   Measures the similarity between two sets. For text, it's often applied to sets of words (tokens).
    *   Formula: `J(A, B) = |A intersect B| / |A union B|`
    *   **Strengths:** Simple, intuitive, good for short texts or when word presence is key.
    *   **Limitations:** Doesn't consider word order or semantic meaning.

2.  **Word Mover's Distance (WMD):**
    *   A metric that measures the dissimilarity between two text documents as the minimum cumulative distance that words from one document need to "travel" to reach words in the other document.
    *   Uses word embeddings (like Word2Vec) to calculate the distance between individual words.
    *   **Strengths:** Captures semantic meaning, robust to word choice variations.
    *   **Limitations:** Computationally more expensive than simpler methods.

3.  **Sentence Embeddings & Similarity:**
    *   Representing entire sentences or paragraphs as single dense vectors.
    *   Models like Sentence-BERT (SBERT) are specifically trained to produce semantically meaningful sentence embeddings.
    *   Similarity is then calculated using cosine similarity between these sentence vectors.
    *   **Strengths:** Captures sentence-level semantics, efficient for large-scale comparisons.

## Task 1: Jaccard Similarity Implementation

1.  **Create the file:** `src/tasks/text_similarity.py`.

2.  **Implement the `TextSimilarity` class:**
    *   The constructor `__init__(self, tokenizer: Tokenizer, word_embedder: WordEmbedder = None)` should accept a `Tokenizer` and optionally a `WordEmbedder` (for WMD).

3.  **Implement `calculate_jaccard_similarity(self, text1: str, text2: str) -> float`:**
    *   Tokenize both `text1` and `text2`.
    *   Convert the token lists into sets.
    *   Calculate the Jaccard similarity using the formula.

## Task 2: Word Mover's Distance (WMD) Implementation

1.  **Implement `calculate_wmd(self, text1: str, text2: str) -> float`:**
    *   This method requires a `WordEmbedder` instance to be passed to the constructor.
    *   Tokenize both `text1` and `text2`.
    *   Use the `WordEmbedder`'s underlying `gensim` model to calculate WMD between the tokenized texts.
    *   Handle cases where words are out of vocabulary for the embedder.

## Task 3: Sentence Similarity (using Sentence-BERT)

1.  **Implement `calculate_sentence_similarity(self, sentence1: str, sentence2: str) -> float`:**
    *   This task requires a pre-trained Sentence-BERT model from `transformers`.
    *   Initialize a `SentenceTransformer` model (e.g., `'all-MiniLM-L6-v2'`) from the `sentence_transformers` library (which builds on `transformers`). You might need to add `sentence-transformers` to `requirements.txt`.
    *   Get embeddings for both `sentence1` and `sentence2`.
    *   Calculate cosine similarity between the two sentence embeddings.

## Evaluation

*   Create a new test file: `test/lab16_test.py`.
*   Instantiate your `RegexTokenizer` and `WordEmbedder` (for WMD).
*   Instantiate your `TextSimilarity` class.
*   Test each implemented method with the following pairs of sentences and print the results:
    *   **Pair 1 (High Similarity):**
        *   `text1 = "The cat sat on the mat."`
        *   `text2 = "A feline was resting on the rug."`
    *   **Pair 2 (Low Similarity):**
        *   `text1 = "The cat sat on the mat."`
        *   `text2 = "The car drove fast on the highway."`
    *   **Pair 3 (Semantic but different words):**
        *   `text1 = "I love eating delicious apples."`
        *   `text2 = "I enjoy consuming tasty fruit."`
