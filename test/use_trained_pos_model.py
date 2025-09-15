import spacy
import os

def main():
    """Loads the custom-trained POS tagger and uses it to predict tags."""
    # Path to the best model saved by spaCy during training
    model_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/results/pos_model/model-best"

    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please ensure the model has been trained successfully.")
        return

    print(f"--- Loading custom POS tagger from: {model_path} ---")
    try:
        nlp = spacy.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Example sentence to test the tagger
    test_sentence = "The quick brown fox jumps over the lazy dog near the river bank."
    print(f"\n--- Testing on sentence: ---\n'{test_sentence}'")

    # Process the text with the loaded model
    doc = nlp(test_sentence)

    # Print the results
    print("\n--- Predicted POS Tags: ---")
    print(f"{'WORD':<15} | {'POS TAG'}")
    print("-" * 25)
    for token in doc:
        print(f"{token.text:<15} | {token.tag_}")

if __name__ == "__main__":
    main()
