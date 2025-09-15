import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tasks.named_entity_recognition import NamedEntityRecognizer

def main():
    """
    Main function to test the NamedEntityRecognizer.
    """
    print("--- NamedEntityRecognizer Evaluation ---")

    # Instantiate the recognizer (will attempt to download model if not present)
    try:
        ner = NamedEntityRecognizer()
    except Exception as e:
        print(f"Error initializing NamedEntityRecognizer: {e}")
        print("Please ensure spaCy and the 'en_core_web_sm' model are installed.")
        print("You can install the model by running: python -m spacy download en_core_web_sm")
        return

    # Test sentences
    test_sentences = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "Dr. Smith works at Google in New York since January 1st, 2023.",
    ]

    for i, sentence in enumerate(test_sentences):
        print(f"\n--- Sentence {i+1} ---")
        print(f"Original: '{sentence}'")
        entities = ner.recognize_entities(sentence)
        print("Recognized Entities:")
        pprint.pprint(entities)

if __name__ == "__main__":
    main()
