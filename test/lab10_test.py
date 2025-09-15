import sys
import os
import pprint
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llms.llm_finetuner import LLMFinetuner

def main():
    """
    Main function to test the LLMFinetuner.
    """
    print("--- LLMFinetuner Evaluation ---")
    print("NOTE: This script will download a pre-trained model and fine-tune it.")
    print("This will take significant time and computational resources.")

    # 1. Define the dataset (same as Lab 5)
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

    print("\nOriginal Data:")
    for i, text in enumerate(texts):
        print(f"  '{text}' -> {labels[i]}")

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    # Further split X_train for a small evaluation set for the Trainer
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42, stratify=y_train
    )


    print("\nTraining Data:")
    for i, text in enumerate(X_train):
        print(f"  '{text}' -> {y_train[i]}")
    print("\nEvaluation Data (for Trainer):")
    for i, text in enumerate(X_eval):
        print(f"  '{text}' -> {y_eval[i]}")
    print("\nTest Data:")
    for i, text in enumerate(X_test):
        print(f"  '{text}' -> {y_test[i]}")

    try:
        # 3. Instantiate LLMFinetuner
        finetuner = LLMFinetuner(model_name="distilbert-base-uncased", num_labels=2)

        # 4. Fine-tune the model
        print("\n--- Starting Fine-tuning ---")
        finetuner.fine_tune(X_train, y_train, X_eval, y_eval)
        print("--- Fine-tuning Complete ---")

        # 5. Evaluate the model
        print("\n--- Evaluation on Test Set ---")
        metrics = finetuner.evaluate_model(X_test, y_test)
        pprint.pprint(metrics)

        # 6. Make a prediction on a new sentence
        print("\n--- Prediction on New Sentence ---")
        new_sentence = "This is an amazing product, I love it!"
        prediction = finetuner.predict([new_sentence])
        print(f"New sentence: '{new_sentence}' -> Predicted label: {prediction[0]}")

    except Exception as e:
        print(f"\nAn error occurred during fine-tuning or evaluation: {e}")
        print("Please ensure all dependencies (transformers, datasets, scikit-learn, torch) are installed.")
        print("Also, check your internet connection for model downloads.")


if __name__ == "__main__":
    main()
