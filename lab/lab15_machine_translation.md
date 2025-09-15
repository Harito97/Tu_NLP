# Lab 15: Machine Translation with Sequence-to-Sequence Models

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab uses pre-trained translation models from Hugging Face. The model (`Helsinki-NLP/opus-mt-en-fr` by default) will be downloaded automatically the first time it is used. This is a large file and the download process may take some time and consume considerable data and disk space.

For optimal performance and to enable BLEU score calculation, it is recommended to install `sacremoses` and `rouge_score`:

```bash
pip install sacremoses rouge_score
```

## Objective

To understand the principles of machine translation and learn to use pre-trained sequence-to-sequence models for translating text between languages.

## Theory

**Machine Translation (MT)** is the task of automatically converting text or speech from one natural language (the source language) into another (the target language).

**Evolution of MT:**
*   **Rule-based MT (RBMT):** Relied on linguistic rules and dictionaries.
*   **Statistical MT (SMT):** Used statistical models trained on parallel corpora.
*   **Neural MT (NMT):** Current state-of-the-art, using neural networks, primarily sequence-to-sequence models.

**Sequence-to-Sequence (Seq2Seq) Models:**
*   A general architecture for tasks that involve mapping an input sequence to an output sequence.
*   Consists of an **Encoder** (reads the source sequence and compresses it into a fixed-length context vector) and a **Decoder** (generates the target sequence from the context vector).
*   **Attention Mechanism:** A crucial component that allows the decoder to focus on different parts of the source sequence at each step of generating the target sequence.

**Hugging Face Transformers for MT:** The `transformers` library provides easy access to many pre-trained NMT models, often fine-tuned on specific language pairs (e.g., `Helsinki-NLP/opus-mt-en-fr` for English to French).

**Evaluation:** The **BLEU (Bilingual Evaluation Understudy)** score (recap from Lab 12) is a widely used metric for evaluating the quality of machine-translated text.

## Task 1: `Translator` Implementation

1.  **Create the file:** `src/tasks/machine_translation.py`.

2.  **Implement the `Translator` class:**
    *   The constructor `__init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-fr")` should initialize a `transformers.pipeline` for translation.

3.  **Implement `translate(self, text: str) -> str`:**
    *   Use the translation pipeline to translate a single input `text`.

4.  **Implement `translate_batch(self, texts: List[str]) -> List[str]`:**
    *   Use the translation pipeline to translate a list of input `texts`.

## Evaluation

*   Create a new test file: `test/lab15_test.py`.
*   Instantiate your `Translator` class.
*   Provide sample English sentences and translate them to French (or another language supported by your chosen model).
*   Print the original and translated texts.
*   (Bonus) If you have reference translations, calculate the BLEU score using your `LLMEvaluator` from Lab 12.
