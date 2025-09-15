# Lab 4: Word Embeddings with Word2Vec

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab uses the `gensim` library for word embeddings. The `WordEmbedder` class will automatically download the `glove-wiki-gigaword-50` model the first time it is used. This download is approximately 65MB and may take some time depending on your internet connection.

## Objective

To move from sparse, high-dimensional representations (like TF-IDF) to dense, low-dimensional semantic representations. You will learn to load and use pre-trained word embeddings to find semantic similarities between words and to create document embeddings.

## Theory

Word embeddings are dense vectors that represent words in a way that captures their meaning, context, and semantic relationships. Unlike TF-IDF, where similar words are treated as completely different features, word embeddings place similar words close to each other in the vector space.

Word2Vec is a popular model used to learn these embeddings from large text corpora. It works on the principle that "a word is characterized by the company it keeps."

For this lab, we will use a pre-trained model, which is a common and effective approach.

## Task 1: Setup

1.  **Install `gensim`:** This library provides easy access to word embedding models. Add `gensim` to your `requirements.txt` file and install it.
2.  **Download Pre-trained Model:** We will use a pre-trained model from `gensim`'s data repository. The model `glove-wiki-gigaword-50` is a good starting point (50-dimensional vectors trained on Wikipedia).

## Task 2: Word Embedding Exploration

1.  **Create the file:** `src/representations/word_embedder.py`.

2.  **Implement the `WordEmbedder` class:**
    *   The constructor `__init__(self, model_name: str)` should accept a model name (e.g., `'glove-wiki-gigaword-50'`).
    *   Inside the constructor, use `gensim.downloader.load(model_name)` to load the model and store it in an attribute.

3.  **Implement exploration methods:**
    *   `get_vector(self, word: str)`: Returns the embedding vector for a given word. Handle cases where the word is not in the vocabulary (Out-of-Vocabulary or OOV words).
    *   `get_similarity(self, word1: str, word2: str)`: Returns the cosine similarity between the vectors of two words.
    *   `get_most_similar(self, word: str, top_n: int = 10)`: Uses the model's built-in `most_similar` method to find the top N most similar words.

## Task 3: Document Embedding

Representing a whole document can be done in many ways. A simple but effective baseline is to average the word vectors of all the words in the document.

1.  **Implement `embed_document(self, document: str)`:**
    *   This method should take a string document as input.
    *   Use a `Tokenizer` (from Lab 1) to split the document into tokens.
    *   For each token, get its vector. Ignore OOV words.
    *   If the document contains no known words, return a zero vector of the correct dimension.
    *   Otherwise, compute the element-wise mean of all the word vectors to get a single document vector.

## Evaluation

*   Create a new test file: `test/lab4_test.py`.
*   Instantiate your `WordEmbedder`.
*   Perform and print the results of the following operations:
    *   Get the vector for the word 'king'.
    *   Get the similarity between 'king' and 'queen', and between 'king' and 'man'.
    *   Get the 10 most similar words to 'computer'.
    *   Embed the sentence "The queen rules the country." and print the resulting document vector.

---

## Bonus Task: Training a Word2Vec Model from Scratch

While using pre-trained models is very effective, it's also insightful to train your own embeddings on a specific domain. The script `test/lab4_embedding_training_demo.py` was created to demonstrate this process.

This script performs the following steps:
1.  **Streams Data**: It reads the raw text from `data/UD_English-EWT/en_ewt-ud-train.txt` in a memory-efficient way.
2.  **Trains a Model**: It uses `gensim` to train a new `Word2Vec` model on this data.
3.  **Saves the Model**: The resulting trained model is saved to `results/word2vec_ewt.model`.
4.  **Demonstrates Usage**: It shows how to use the newly trained model to find similar words and solve analogies, providing a complete end-to-end example.

To run this demonstration, execute the following command from the project root:
```bash
python test/lab4_embedding_training_demo.py
```
