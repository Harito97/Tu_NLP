# Lab 8: Language Models - N-gram Models

## Objective

To understand the fundamental concept of language modeling, its applications, and to implement a simple N-gram language model.

## Theory

A **Language Model (LM)** is a probability distribution over sequences of words. It assigns a probability to a sequence of words, or, more commonly, predicts the probability of the next word in a sequence given the preceding words.

**Applications:**
*   Speech Recognition
*   Machine Translation
*   Text Generation
*   Spelling Correction

**N-gram Models:** These models make a simplifying assumption (the Markov Assumption) that the probability of a word depends only on the previous `N-1` words.

*   **Unigram (N=1):** `P(word)`
*   **Bigram (N=2):** `P(word_n | word_{n-1})`
*   **Trigram (N=3):** `P(word_n | word_{n-2}, word_{n-1})`

**Maximum Likelihood Estimation (MLE):** Probabilities are estimated by counting occurrences in a training corpus.

*   `P(word_n | word_{n-1}) = Count(word_{n-1} word_n) / Count(word_{n-1})`

**Smoothing:** To handle N-grams that don't appear in the training corpus (which would result in zero probabilities), smoothing techniques are used (e.g., Add-one smoothing, Kneser-Ney smoothing). For this lab, we'll implement a basic version that returns 0 for unseen N-grams, or you can implement a simple Add-one smoothing as a bonus.

## Task 1: `NgramLanguageModel` Implementation

1.  **Create the file:** `src/models/ngram_language_model.py`.

2.  **Implement the `NgramLanguageModel` class:**
    *   The constructor `__init__(self, tokenizer: Tokenizer, n: int = 2)` should accept a `Tokenizer` instance and the `n` value for the N-gram (e.g., 2 for bigram, 3 for trigram).
    *   It should store counts for N-grams and (N-1)-grams.

3.  **Implement `fit(self, corpus: List[str])`:**
    *   Tokenize each document in the corpus.
    *   For each document, generate all N-grams and (N-1)-grams.
    *   Count their occurrences and store them (e.g., in dictionaries).
    *   Consider adding special `<s>` (start-of-sentence) and `</s>` (end-of-sentence) tokens to handle sentence boundaries, especially for bigrams and trigrams.

4.  **Implement `predict_next_word(self, context: List[str]) -> Dict[str, float]`:**
    *   Given a `context` (a list of previous words, typically `n-1` words long), predict the probability distribution of the next word.
    *   Extract the relevant `(n-1)`-gram from the context.
    *   Calculate `P(word | context)` for all possible next words based on your stored counts.
    *   Return a dictionary mapping each possible next word to its probability.
    *   Handle unseen N-grams (e.g., return 0 probability for now, or implement Add-one smoothing as a bonus).

## Task 2: Text Generation (Bonus)

1.  **Implement `generate_text(self, seed_text: str, length: int = 20) -> str`:**
    *   Given a `seed_text` (e.g., "The quick brown"), generate a sequence of `length` words.
    *   Use `predict_next_word` to sample the next word based on its probability distribution.
    *   Continuously update the context and generate words until the desired length is reached or an `</s>` token is generated.

## Evaluation

*   Create a new test file: `test/lab8_test.py`.
*   Define a small corpus (e.g., a few sentences).
*   Instantiate your `RegexTokenizer`.
*   Instantiate your `NgramLanguageModel` (e.g., with `n=2` for bigrams).
*   Train the model on your corpus.
*   Test `predict_next_word` with a few contexts and print the probability distributions.
*   (Bonus) Test `generate_text` with a seed and print the generated text.

## Task 3: Training with UD_English-EWT Dataset

1.  **Load Dataset:**
    *   In your test file (`test/lab8_test.py` or a new script), import `load_raw_text_data` from `src.core.dataset_loaders`.
    *   Load the raw text from the `UD_English-EWT` dataset:
        ```python
        from src.models.ngram_language_model import NgramLanguageModel
        from src.preprocessing.regex_tokenizer import RegexTokenizer
        from src.core.dataset_loaders import load_raw_text_data

        # ... (your tokenizer and model instantiations) ...

        dataset_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-train.txt"
        raw_text = load_raw_text_data(dataset_path)
        
        # For N-gram models, it's often better to process sentence by sentence
        # Simple split by newline for demonstration, more robust parsing might be needed
        corpus_sentences = raw_text.split('\n')
        
        # Instantiate a tokenizer (e.g., RegexTokenizer)
        tokenizer = RegexTokenizer()
        
        # Instantiate an N-gram Language Model (e.g., bigram)
        ngram_model = NgramLanguageModel(tokenizer=tokenizer, n=2)
        
        print("\n--- Training N-gram Model with UD_English-EWT ---")
        ngram_model.fit(corpus_sentences)
        print("Model training complete.")
        
        # Test prediction
        context = ["The", "quick"]
        print(f"\nPredicting next word for context {context}:")
        predictions = ngram_model.predict_next_word(context)
        print(predictions)
        
        # Test text generation
        print("\nGenerating text:")
        generated_text = ngram_model.generate_text(seed_text="The quick brown", length=10)
        print(generated_text)
        ```
    *   Observe how the model performs with a larger, more realistic corpus. Compare the quality of predictions and generated text to when you used a small, custom corpus.
