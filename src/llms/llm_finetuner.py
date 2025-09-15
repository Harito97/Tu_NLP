import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

class LLMFinetuner:
    """
    A class to fine-tune a pre-trained Transformer-based LLM for text classification.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2):
        """
        Initializes the LLMFinetuner with a pre-trained model and tokenizer.

        Args:
            model_name: The name of the pre-trained model to use.
            num_labels: The number of output labels for classification.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.trainer = None # Will be initialized during fine-tuning

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length")

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def fine_tune(self, train_texts: List[str], train_labels: List[int], eval_texts: List[str], eval_labels: List[int]) -> None:
        """
        Fine-tunes the LLM on the provided training and evaluation data.

        Args:
            train_texts: List of training text documents.
            train_labels: List of corresponding training labels.
            eval_texts: List of evaluation text documents.
            eval_labels: List of corresponding evaluation labels.
        """
        # Create Hugging Face Dataset objects
        train_dataset_dict = {"text": train_texts, "label": train_labels}
        train_dataset = Dataset.from_dict(train_dataset_dict)
        
        eval_dataset_dict = {"text": eval_texts, "label": eval_labels}
        eval_dataset = Dataset.from_dict(eval_dataset_dict)

        # Tokenize datasets
        tokenized_train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self._tokenize_function, batched=True)

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none", # Disable reporting to services like Weights & Biases
        )

        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        # Train the model
        self.trainer.train()

    def evaluate_model(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, float]:
        """
        Evaluates the fine-tuned model on a test set.

        Args:
            test_texts: List of test text documents.
            test_labels: List of corresponding test labels.

        Returns:
            A dictionary containing evaluation metrics.
        """
        if self.trainer is None:
            raise RuntimeError("Model has not been fine-tuned yet. Call fine_tune() first.")
        
        test_dataset_dict = {"text": test_texts, "label": test_labels}
        test_dataset = Dataset.from_dict(test_dataset_dict)
        tokenized_test_dataset = test_dataset.map(self._tokenize_function, batched=True)

        metrics = self.trainer.evaluate(tokenized_test_dataset)
        return metrics

    def predict(self, texts: List[str]) -> List[int]:
        """
        Makes predictions on new texts using the fine-tuned model.

        Args:
            texts: List of text documents to predict.

        Returns:
            A list of predicted integer labels.
        """
        if self.trainer is None:
            raise RuntimeError("Model has not been fine-tuned yet. Call fine_tune() first.")
        
        predict_dataset_dict = {"text": texts, "label": [0]*len(texts)} # Dummy labels
        predict_dataset = Dataset.from_dict(predict_dataset_dict)
        tokenized_predict_dataset = predict_dataset.map(self._tokenize_function, batched=True)

        predictions = self.trainer.predict(tokenized_predict_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=-1).tolist()
        return predicted_labels
