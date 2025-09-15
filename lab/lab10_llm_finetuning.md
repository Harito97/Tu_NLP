# Lab 10: Fine-tuning LLMs for Text Classification

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab requires the `accelerate` library for efficient fine-tuning with Hugging Face Transformers. Install it separately if you encounter issues:

```bash
pip install accelerate
```

**IMPORTANT NOTE:** Fine-tuning Large Language Models is a computationally intensive task. This lab will download a pre-trained model and then fine-tune it, which requires significant time, CPU/RAM, and potentially GPU resources. Ensure you have a stable internet connection and sufficient resources before running the evaluation script.

## Objective

To learn how to adapt a pre-trained Large Language Model (LLM) to a specific text classification task by fine-tuning it on a custom dataset. This demonstrates how to leverage the powerful general language understanding of LLMs for specialized applications.

## Theory

**Fine-tuning** is the process of taking a pre-trained model (which has learned general language patterns from a massive corpus) and further training it on a smaller, task-specific dataset. This allows the model to specialize in a particular task or domain while retaining its broad language understanding.

**Why Fine-tune?**
*   **Domain Adaptation:** Improve performance on text from a specific domain (e.g., legal, medical).
*   **Task Specialization:** Adapt a general-purpose LLM to a specific task (e.g., sentiment analysis, spam detection).
*   **Better Performance:** Often yields superior results compared to training a model from scratch on small datasets.

**Fine-tuning Process:**
1.  **Choose a Pre-trained Model:** Select a suitable model (e.g., BERT, RoBERTa, DistilBERT) and its corresponding tokenizer.
2.  **Prepare Dataset:** Tokenize your task-specific dataset using the model's tokenizer and format it for training.
3.  **Define Training Arguments:** Set hyperparameters like learning rate, batch size, number of epochs.
4.  **Train the Model:** Use the prepared dataset and arguments to fine-tune the model.
5.  **Evaluate:** Assess the model's performance on a held-out test set.

We will use the `Hugging Face Transformers` library, specifically its `Trainer` API, which simplifies the fine-tuning process.

## Task 1: Dataset Preparation

1.  **Dataset:** We'll use a small, custom sentiment dataset similar to Lab 5.
    ```python
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative
    ```
2.  **Tokenization:** Use a `transformers` tokenizer (e.g., `AutoTokenizer.from_pretrained("distilbert-base-uncased")`) to tokenize these texts.
3.  **Dataset Object:** Convert the tokenized data into a `torch.utils.data.Dataset` or `transformers.Dataset` object suitable for the `Trainer`.

## Task 2: Model Fine-tuning

1.  **Create the file:** `src/llms/llm_finetuner.py`.

2.  **Implement the `LLMFinetuner` class:**
    *   The constructor `__init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2)` should load the pre-trained model and tokenizer.
    *   Implement `fine_tune(self, train_texts: List[str], train_labels: List[int], eval_texts: List[str], eval_labels: List[int]) -> None`:
        *   Prepare the training and evaluation datasets.
        *   Define `TrainingArguments` (e.g., output directory, learning rate, epochs).
        *   Create a `Trainer` instance.
        *   Train the model using `trainer.train()`.

## Task 3: Evaluation and Prediction

1.  **Implement `evaluate_model(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, float]`:**
    *   This method will evaluate the fine-tuned model on a test set.
    *   Use `trainer.evaluate()` to get metrics.

2.  **Implement `predict(self, texts: List[str]) -> List[int]`:**
    *   This method will make predictions on new texts using the fine-tuned model.

## Evaluation

*   Create a new test file: `test/lab10_test.py`.
*   Define the `texts` and `labels` dataset.
*   Split the data into training and testing sets (e.g., 80% train, 20% test) using `sklearn.model_selection.train_test_split`.
*   Instantiate your `LLMFinetuner`.
*   Call `fine_tune` with your training and evaluation data.
*   Call `evaluate_model` with your test data and print the metrics.
*   Call `predict` on a new sentence and print the prediction.
