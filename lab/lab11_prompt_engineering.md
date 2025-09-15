# Lab 11: Prompt Engineering and Advanced LLM Interaction

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab uses a pre-trained Large Language Model (`gpt2`) from Hugging Face. The model will be downloaded automatically the first time it is used. This is a large file and the download process may take some time and consume considerable data and disk space.

## Objective

To explore advanced techniques for interacting with Large Language Models (LLMs) through prompt engineering, enabling them to perform complex tasks, reason, and generate specific types of output without explicit fine-tuning.

## Theory

**Prompt Engineering** is the art and science of crafting effective inputs (prompts) to guide LLMs to generate desired outputs. It's a crucial skill for leveraging the full potential of LLMs.

**Key Concepts:**
*   **Zero-shot Prompting:** Asking the LLM to perform a task without any examples.
*   **One-shot Prompting:** Providing one example in the prompt to guide the LLM.
*   **Few-shot Prompting:** Providing a few examples in the prompt to guide the LLM.
*   **Chain-of-Thought (CoT) Prompting:** Encouraging the LLM to show its reasoning steps, which can lead to more accurate results for complex problems.
*   **Role-playing/Persona Prompting:** Instructing the LLM to adopt a specific persona or role.
*   **Delimiters:** Using special characters (e.g., `###`, `"""`) to clearly separate instructions from context.

## Task 1: Few-shot Text Classification

1.  **Create the file:** `src/llms/prompt_engineer.py`.

2.  **Implement the `PromptEngineer` class:**
    *   The constructor `__init__(self, model_name: str = "gpt2")` should initialize a `transformers.pipeline` for text generation.

3.  **Implement `few_shot_classify(self, text: str, examples: List[Dict[str, str]]) -> str`:**
    *   Construct a prompt that includes the `examples` (e.g., `[{"text": "I love it", "label": "positive"}]`) and then asks the LLM to classify the `text`.
    *   Use the text generation pipeline to get the classification.

## Task 2: Chain-of-Thought Reasoning

1.  **Implement `reason_with_cot(self, problem: str) -> str`:**
    *   Design a prompt that encourages the LLM to think step-by-step to solve a `problem` (e.g., a simple math word problem or a logical puzzle).
    *   The prompt should explicitly ask for reasoning before the final answer.

## Task 3: Role-playing/Persona Prompting

1.  **Implement `role_play(self, role: str, query: str) -> str`:**
    *   Design a prompt that instructs the LLM to adopt a specific `role` (e.g., "a grumpy old man", "a helpful customer service agent") and respond to a `query`.

## Evaluation

*   Create a new test file: `test/lab11_test.py`.
*   Instantiate your `PromptEngineer` class.
*   Test each implemented method with appropriate inputs and print the results.
    *   For few-shot classification, use a new sentence and the provided examples.
    *   For CoT, use a simple reasoning problem.
    *   For role-playing, use a role and a query.
