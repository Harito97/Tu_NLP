# Lab 9: Large Language Models (LLMs)

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab uses pre-trained Large Language Models from Hugging Face. The models (`gpt2` for text generation, `t5-small` for summarization, and `distilbert-base-uncased-distilled-squad` for question answering) will be downloaded automatically the first time they are used. These are large files and the download process may take a significant amount of time and consume considerable data and disk space.

## Objective

To understand the fundamental concepts behind Large Language Models (LLMs) and learn how to leverage pre-trained transformer-based models for advanced NLP tasks such as text generation, summarization, and question answering.

## Theory

**Large Language Models (LLMs)** are advanced neural networks, typically based on the Transformer architecture, trained on vast amounts of text data (trillions of words). This massive scale allows them to learn complex patterns, grammar, facts, and even reasoning abilities.

**Key Characteristics:**
*   **Transformer Architecture:** Relies heavily on the self-attention mechanism, allowing models to weigh the importance of different words in a sequence.
*   **Pre-training:** Models are first trained on a general corpus to learn language understanding.
*   **Fine-tuning / Prompt Engineering:** Pre-trained models can be adapted to specific tasks (fine-tuning) or guided with carefully crafted prompts (zero-shot/few-shot learning).

**Common LLM Tasks:**
*   Text Generation (e.g., writing articles, creative content)
*   Summarization (e.g., condensing long documents)
*   Question Answering (e.g., extracting answers from text)
*   Translation
*   Code Generation

We will use the **Hugging Face Transformers** library, which provides a unified API for accessing and using state-of-the-art pre-trained models.

## Task 1: Text Generation

1.  **Create the file:** `src/llms/llm_tasks.py`.

2.  **Implement a `LLMTasks` class:**
    *   The constructor `__init__(self)` can initialize a `pipeline` for text generation.

3.  **Implement `generate_text(self, prompt: str, max_length: int = 50) -> str`:**
    *   Use `transformers.pipeline("text-generation", model="gpt2")`.
    *   Generate text based on the `prompt` and `max_length`.

## Task 2: Text Summarization

1.  **Implement `summarize_text(self, text: str, max_length: int = 100, min_length: int = 30) -> str`:**
    *   Use `transformers.pipeline("summarization", model="t5-small")`.
    *   Summarize the given `text`.

## Task 3: Question Answering

1.  **Implement `answer_question(self, question: str, context: str) -> Dict[str, str]`:**
    *   Use `transformers.pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")`.
    *   Provide a `question` and a `context`.
    *   Return the answer and potentially other information (e.g., score, start/end).

## Evaluation

*   Create a new test file: `test/lab9_test.py`.
*   Instantiate your `LLMTasks` class.
*   Test each implemented method with appropriate prompts/texts/questions and print the results.
    *   For text generation, use a prompt like "The future of AI is".
    *   For summarization, use a paragraph of text.
    *   For question answering, use a context and a question related to it.
