import sys
import os
import pprint
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.tfidf_vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from src.pipelines.text_pipeline import TextPipeline

def main():
    """
    Main function to test the TextPipeline.
    """
    print("--- TextPipeline Evaluation ---")

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

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # 3. Instantiate tokenizer, vectorizer, and classifier
    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    classifier = TextClassifier(vectorizer=vectorizer)

    # 4. Train the classifier (this also fits the vectorizer internally)
    classifier.fit(X_train, y_train)
    print("\nClassifier trained and vectorizer fitted.")

    # 5. Instantiate the TextPipeline
    pipeline = TextPipeline(tokenizer=tokenizer, vectorizer=vectorizer, classifier=classifier)
    print("\nTextPipeline instantiated.")

    # 6. Test process method with a single new sentence
    single_sentence = "This is a great film!"
    single_prediction = pipeline.process(single_sentence)
    print(f"\nSingle sentence: '{single_sentence}' -> Predicted: {single_prediction}")

    # 7. Test process_batch method with the test set
    batch_predictions = pipeline.process_batch(X_test)
    print("\nBatch predictions on Test Data:")
    for i, text in enumerate(X_test):
        print(f"  '{text}' -> Predicted: {batch_predictions[i]}, Actual: {y_test[i]}")

    # 8. Evaluate the batch predictions (optional, but good for completeness)
    metrics = classifier.evaluate(y_test, batch_predictions)
    print("\nEvaluation Metrics for Batch Predictions:")
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
