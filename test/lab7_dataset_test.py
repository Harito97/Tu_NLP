import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.dataset_loaders import load_conllu_data

def main():
    """
    Main function to demonstrate loading CoNLL-U data and inspecting
    token-level classifications (Part-of-Speech tags).
    """
    print("--- Demonstrating CoNLL-U Data Loading for Token Classification (Lab 7) ---")

    # 1. Define the path to the dataset
    dataset_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-dev.conllu"

    # 2. Check if the dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # 3. Load the CoNLL-U data using the loader from 'src'
    try:
        sentences = load_conllu_data(dataset_path)
        print(f"\nSuccessfully loaded {len(sentences)} sentences.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    # 4. Inspect the first 2 sentences to see token classifications
    print("\n--- Inspecting first 2 sentences ---")
    for i, sentence in enumerate(sentences[:2]):
        print(f"\n--- Sentence {i+1} ---")
        if 'text' in sentence.metadata:
            print(f"Text: {sentence.metadata['text']}")
        
        print("Tokens and their UPOS (Universal Part-of-Speech) tags:")
        for token in sentence:
            # Each 'token' is a dictionary-like object with fields from the CoNLL-U format
            token_id = token['id']
            word = token['form']
            pos_tag = token['upos']
            print(f"  - ID: {token_id}, Word: '{word}', POS: {pos_tag}")

if __name__ == "__main__":
    main()
