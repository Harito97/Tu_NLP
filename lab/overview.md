# Overview of Natural Language Processing Labs

This document provides a brief overview of each lab in the NLP course, outlining their objectives, key content, and prerequisites.

---

## Lab 1: Text Tokenization

- **Objective**: Understand and implement fundamental text tokenization techniques.
- **Content**:
  - Introduction to tokenization: Why it's crucial for NLP.
  - Implementing a simple rule-based tokenizer.
  - Implementing a regex-based tokenizer for more robust tokenization.
  - Handling punctuation and special characters.
  - Using the UD_English-EWT dataset for practical tokenization exercises.
- **Builds on**: Basic Python programming skills.

---

## Lab 2: Count Vectorization

- **Objective**: Learn to represent text as numerical vectors using word counts.
- **Content**:
  - Introduction to Bag-of-Words (BoW) model.
  - Implementing a custom CountVectorizer.
  - Vocabulary creation and mapping words to indices.
  - Sparse matrix representation for efficiency.
- **Builds on**: Lab 1 (Tokenization).

---

## Lab 3: TF-IDF Vectorization

- **Objective**: Understand and implement TF-IDF (Term Frequency-Inverse Document Frequency) for weighting word importance in a corpus.
- **Content**:
  - Understanding Term Frequency (TF) and Inverse Document Frequency (IDF).
  - Implementing a custom TF-IDF Vectorizer.
  - Comparing TF-IDF with simple Count Vectorization.
  - Applications in information retrieval and text mining.
- **Builds on**: Lab 2 (Count Vectorization).

---

## Lab 4: Word Embeddings

- **Objective**: Explore dense vector representations of words and their semantic relationships.
- **Content**:
  - Introduction to the concept of word embeddings (e.g., Word2Vec, GloVe, FastText).
  - Using pre-trained word embeddings (e.g., from spaCy or Gensim).
  - Calculating semantic similarity between words.
  - Visualizing word embeddings (e.g., using t-SNE or PCA).
- **Builds on**: Lab 3 (conceptual understanding of vectorization).

---

## Lab 5: Text Classification

- **Objective**: Implement text classification using traditional machine learning models.
- **Content**:
  - Overview of text classification tasks (e.g., sentiment analysis, spam detection).
  - Feature extraction using Count and TF-IDF vectors.
  - Training and evaluating classifiers (e.g., Naive Bayes, Support Vector Machines).
  - Understanding common evaluation metrics (accuracy, precision, recall, F1-score).
- **Builds on**: Lab 2 (Count Vectorization), Lab 3 (TF-IDF Vectorization).

---

## Lab 6: NLP Pipeline

- **Objective**: Build an end-to-end Natural Language Processing pipeline by combining various preprocessing, vectorization, and classification steps.
- **Content**:
  - Designing a modular NLP pipeline.
  - Integrating tokenization, vectorization, and classification components.
  - Data flow and transformation between pipeline stages.
  - Evaluating the performance of the complete pipeline.
- **Builds on**: Lab 1 (Tokenization), Lab 2 (Count Vectorization), Lab 3 (TF-IDF Vectorization), Lab 5 (Text Classification).

---

## Lab 7: Token Classification - Named Entity Recognition (NER)

- **Objective**: Understand and implement a fundamental token classification task: Named Entity Recognition (NER).
- **Content**:
  - Introduction to token classification and NER.
  - Using the `spaCy` library for pre-trained NER models.
  - Extracting entities (PERSON, ORG, LOC, etc.) from text.
  - Loading and preparing CoNLL-U formatted data from the UD_English-EWT dataset for token-level tasks.
- **Builds on**: Lab 1 (Tokenization).

---

## Lab 8: Language Models - N-gram Models

- **Objective**: Understand the fundamental concept of language modeling and implement a simple N-gram language model.
- **Content**:
  - Introduction to language models and their applications.
  - N-gram probabilities and Maximum Likelihood Estimation.
  - Implementing an N-gram language model (e.g., bigram, trigram).
  - Basic smoothing techniques (e.g., Add-one smoothing).
  - Text generation using the N-gram model.
  - Training the model using the UD_English-EWT dataset.
- **Builds on**: Lab 1 (Tokenization).

---

## Lab 9: Large Language Models (LLMs)

- **Objective**: Introduce modern Large Language Models (LLMs) and their basic applications.
- **Content**:
  - High-level overview of the Transformer architecture.
  - Using pre-trained LLMs from Hugging Face Transformers.
  - Basic prompting techniques for various tasks (e.g., summarization, question answering).
  - Understanding the capabilities and limitations of LLMs.
- **Builds on**: Lab 4 (conceptual understanding of embeddings).

---

## Lab 10: LLM Fine-tuning

- **Objective**: Learn how to fine-tune Large Language Models for specific downstream tasks.
- **Content**:
  - Introduction to transfer learning in the context of LLMs.
  - Preparing datasets for fine-tuning.
  - Using Hugging Face `transformers` library for fine-tuning a pre-trained LLM on a custom dataset.
  - Evaluating fine-tuned models.
- **Builds on**: Lab 9 (Large Language Models).

---

## Lab 11: Prompt Engineering

- **Objective**: Master techniques for designing effective prompts to guide Large Language Models for desired outputs.
- **Content**:
  - Principles of effective prompt design.
  - Zero-shot, few-shot prompting.
  - Chain-of-Thought (CoT) prompting and other advanced techniques.
  - Role-playing and persona-based prompting.
  - Strategies for improving prompt robustness and reducing hallucinations.
- **Builds on**: Lab 9 (Large Language Models).

---

## Lab 12: LLM Evaluation & Ethics

- **Objective**: Understand methods for evaluating Large Language Model performance and discuss critical ethical considerations.
- **Content**:
  - Quantitative evaluation metrics (e.g., BLEU, ROUGE, perplexity - high-level overview).
  - Human evaluation methodologies for LLMs.
  - Discussion of ethical concerns: bias, fairness, toxicity, privacy, misinformation.
  - Strategies for mitigating ethical risks in LLM deployment.
- **Builds on**: Lab 9 (Large Language Models), Lab 10 (LLM Fine-tuning), Lab 11 (Prompt Engineering).

---

## Lab 13: Advanced Text Preprocessing (Stemming, Lemmatization, and POS Tagging)

- Objective: Deepen understanding of text preprocessing by implementing
  stemming and lemmatization, and performing Part-of-Speech (POS) tagging to
  extract more linguistic features from text.
- Content:
  - Stemming: Implementing a rule-based stemmer (e.g., Porter Stemmer) or
    using NLTK.
  - Lemmatization: Using spaCy or NLTK for dictionary-based lemmatization.
  - Part-of-Speech (POS) Tagging: Using spaCy to assign grammatical tags to
    each word in a sentence.
  - Applications: How these techniques improve vectorization and downstream
    NLP tasks.
- Builds on: Lab 1 (Tokenization).

---

## Lab 14: Text Clustering

- Objective: Explore unsupervised learning techniques to group similar text
  documents together based on their content, without requiring predefined
  labels.
- Content:
  - Document Representation: Using TF-IDF vectors (Lab 3) or averaged word
    embeddings (Lab 4) as input features.
  - Clustering Algorithms: Implementing or using scikit-learn for K-Means
    clustering.
  - Evaluation: Discussing metrics for clustering evaluation (e.g.,
    Silhouette Score, Adjusted Rand Index).
  - Interpretation: Analyzing cluster centroids or most representative
    documents to understand the discovered topics.
- Builds on: Lab 2 (Count Vectorization), Lab 3 (TF-IDF), Lab 4 (Word
  Embeddings).

---

## Lab 15: Machine Translation with Sequence-to-Sequence Models

- Objective: Understand the principles of machine translation and learn to use
  pre-trained sequence-to-sequence models for translating text between
  languages.
- Content:
  - Introduction to Sequence-to-Sequence (Seq2Seq): Encoder-decoder
    architecture, the role of attention mechanism (high-level overview).
  - Using Pre-trained Models: Leveraging Hugging Face Transformers for
    machine translation pipelines (e.g., using models like
    Helsinki-NLP/opus-mt-en-fr).
  - Evaluation: Discussing metrics like BLEU score (introduced in Lab 12) for
    translation quality.
  - Multilingual Aspects: Briefly touching upon multilingual models.
- Builds on: Lab 9 (LLMs), Lab 12 (Evaluation).

---

## Lab 16: Advanced Text Similarity

- Objective: Explore different methods for calculating text similarity
  beyond simple cosine similarity of TF-IDF vectors, including Jaccard
  similarity and semantic similarity using word embeddings or sentence
  transformers.
- Content:
  - Jaccard Similarity: For comparing sets of words (e.g., for short
    texts).
  - Word Mover's Distance (WMD): A metric that measures the
    dissimilarity between two text documents as the minimum
    cumulative distance that words from one document need to travel
    to reach words in the other document. (Uses gensim).
  - Semantic Similarity with Sentence Embeddings: Using models like
    Sentence-BERT to get embeddings for entire sentences and then
    calculating cosine similarity.
- Builds on: Lab 2/3 (Vectorization), Lab 4 (Embeddings).

---

## Lab 17: Building a Simple Chatbot (Rule-based/Retrieval-based)

- Objective: Create a basic chatbot that can respond to user queries
  based on predefined rules or by retrieving relevant answers from a
  knowledge base. This integrates several previous concepts into a
  practical application.
- Content:
  - Intent Recognition: Using text classification (Lab 5) to
    determine the user's intent.
  - Entity Extraction: Using NER (Lab 7) to identify key information
    in the user's query.
  - Rule-based Responses: Defining simple if-then rules for specific
    intents and entities.
  - Retrieval-based Q&A: Matching user queries to a database of
    questions and answers (e.g., using vector similarity from Lab
    16).
- Builds on: Lab 5 (Text Classification), Lab 7 (NER), Lab 16 (Text
  Similarity).

---

## Lab 18: Multilingual NLP

- Objective: Understand the challenges and approaches to processing
  and generating text in multiple languages, leveraging multilingual
  models.
- Content:
  - Challenges of Multilingual NLP: Language diversity, resource
    scarcity.
  - Multilingual Models: Using pre-trained multilingual models from
    Hugging Face (e.g., mBERT, XLM-R) that can process text in many
    languages.
  - Zero-shot Cross-lingual Transfer: Applying a model trained on
    one language to a task in another language.
  - Multilingual Embeddings: Exploring how words from different
    languages can be mapped into a shared vector space.
- Builds on: Lab 9 (LLMs), Lab 10 (Fine-tuning), Lab 15 (Translation).

---

## Lab 19: Extractive Summarization

- Objective: Implement a method for extractive summarization, where
  the summary consists of sentences directly extracted from the
  original document. This is a simpler form of summarization compared
  to the abstractive method in Lab 9.
- Content:
  - Sentence Tokenization: Breaking a document into individual
    sentences.
  - Sentence Embedding: Representing each sentence as a vector
    (e.g., by averaging word embeddings from Lab 4, or using
    Sentence-BERT).
  - Sentence Scoring/Clustering: Identifying the most important or
    representative sentences (e.g., based on centrality, TF-IDF
    scores, or clustering sentences and picking centroids).
  - Summary Generation: Selecting and ordering the top-scoring
    sentences.
- Builds on: Lab 1 (Tokenization), Lab 4 (Embeddings), Lab 14
  (Clustering).

---

## Lab 20: Introduction to Speech-to-Text (Conceptual/Using API)

- Objective: Gain a conceptual understanding of Speech-to-Text (STT)
  technology and learn how to use pre-trained STT models or APIs to
  convert spoken language into text.
- Content:
  - STT Pipeline Overview: Audio feature extraction, acoustic model,
    language model.
  - Challenges: Noise, accents, speaker variability.
  - Using Pre-trained Models: Leveraging Hugging Face Transformers
    for STT (e.g., Wav2Vec2, Whisper).
  - Cloud STT APIs: Brief overview of services like Google Cloud
    Speech-to-Text, AWS Transcribe.
  - (Note: Implementing STT from scratch is highly complex and beyond
    a single lab. This lab would focus on usage and conceptual
    understanding.)
- Builds on: General NLP understanding, introduces a new modality.