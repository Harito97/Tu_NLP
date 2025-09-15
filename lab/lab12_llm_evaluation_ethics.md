# Lab 12: LLM Evaluation and Ethical Considerations

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab uses the `evaluate` library for metric calculation. Specifically, for ROUGE scores, you need to install `rouge_score`:

```bash
pip install rouge_score
```

The `evaluate` library might also download additional builder scripts the first time a metric is used.

## Objective

To understand advanced methods for evaluating Large Language Model (LLM) performance, particularly for generative tasks, and to critically analyze the ethical implications, biases, and limitations of LLMs.

## Theory

Evaluating LLMs, especially for open-ended generative tasks, is more complex than traditional classification. We often rely on metrics that compare generated text to reference text, and crucially, on human judgment.

**Evaluation Metrics for Generative Models:**
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Commonly used for summarization. Measures the overlap of n-grams between the generated summary and reference summaries.
*   **BLEU (Bilingual Evaluation Understudy):** Commonly used for machine translation and text generation. Measures the precision of n-grams in the generated text compared to reference texts.
*   **Perplexity:** A measure of how well a probability model predicts a sample. Lower perplexity generally means a better language model.

**Human Evaluation:** Often considered the gold standard for assessing quality, coherence, fluency, and factual correctness, but it is expensive and time-consuming.

**Ethical Considerations:**
*   **Bias:** LLMs can perpetuate and amplify biases present in their training data, leading to unfair or discriminatory outputs.
*   **Fairness:** Ensuring the model performs equally well across different demographic groups.
*   **Toxicity & Harmful Content:** Generating hate speech, misinformation, or other harmful content.
*   **Misinformation/Disinformation:** Generating plausible but false information.
*   **Privacy:** Potential to leak sensitive information from training data.
*   **Environmental Impact:** Training and running large models consume significant energy.

**Mitigating Risks:** Guardrails, red teaming, transparency, explainability, data curation.

## Task 1: ROUGE Score Calculation

1.  **Create the file:** `src/llms/llm_evaluator.py`.

2.  **Implement the `LLMEvaluator` class:**
    *   The constructor `__init__(self)` can initialize evaluators from the `evaluate` library.

3.  **Implement `calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]`:**
    *   Use `evaluate.load("rouge")` to load the ROUGE metric.
    *   Calculate ROUGE scores (e.g., ROUGE-1, ROUGE-2, ROUGE-L) and return them.

## Task 2: BLEU Score Calculation

1.  **Implement `calculate_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]`:**
    *   Use `evaluate.load("bleu")` to load the BLEU metric.
    *   Calculate BLEU score and return it.

## Task 3: Ethical Discussion (Conceptual)

This task is conceptual and will be addressed in the evaluation script.

## Evaluation

*   Create a new test file: `test/lab12_test.py`.
*   Instantiate your `LLMEvaluator` class.
*   **ROUGE:** Provide sample generated summaries and reference summaries. Calculate and print ROUGE scores.
*   **BLEU:** Provide sample generated text and reference texts (for example, a simple translation task). Calculate and print BLEU scores.
*   **Ethical Scenario:** Present a scenario (e.g., an LLM generating biased job descriptions) and discuss the ethical implications and potential mitigation strategies.
