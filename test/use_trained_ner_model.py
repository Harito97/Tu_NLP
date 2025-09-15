import spacy
import os

def main():
    """Loads the custom-trained NER model and uses it to predict entities."""
    # Path to the best model saved by spaCy during training
    model_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/results/ner_conll2003_model/model-best"

    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please ensure the model has been trained successfully.")
        return

    print(f"--- Loading custom NER model from: {model_path} ---")
    try:
        nlp = spacy.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Example sentence to test the NER model
    test_sentence = "U.N. official Ekeus heads for Baghdad to allow Germany and the United States to take part."
    print(f"\n--- Testing on sentence: ---\n'{test_sentence}'")

    # Process the text with the loaded model
    doc = nlp(test_sentence)

    # Print the results
    print("\n--- Predicted Entities: ---")
    if doc.ents:
        for ent in doc.ents:
            print(f"  - Text: '{ent.text}', Label: {ent.label_}")
    else:
        print("No entities found.")

if __name__ == "__main__":
    main()
