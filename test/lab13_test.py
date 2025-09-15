import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.preprocessing.advanced_preprocessor import AdvancedPreprocessor

def main():
    """
    Main function to test the AdvancedPreprocessor.
    """
    print("--- AdvancedPreprocessor Evaluation ---")
    print("NOTE: This script may download NLTK data and spaCy models if not present.")

    # 1. Instantiate tokenizer and preprocessor
    tokenizer = RegexTokenizer()
    try:
        preprocessor = AdvancedPreprocessor(tokenizer=tokenizer)
    except Exception as e:
        print(f"Error initializing AdvancedPreprocessor: {e}")
        print("Please ensure spaCy and NLTK data are installed.")
        print("For spaCy: python -m spacy download en_core_web_sm")
        print("For NLTK: nltk.download('punkt') and nltk.download('wordnet')")
        return

    # 2. Test sentence
    test_sentence = "The quick brown foxes are running quickly to their dens."

    # 3. Test stem_text
    print("\n--- Stemming ---")
    stemmed_tokens = preprocessor.stem_text(test_sentence)
    print(f"Original: '{test_sentence}'")
    print(f"Stemmed: {stemmed_tokens}")

    # 4. Test lemmatize_text
    print("\n--- Lemmatization ---")
    lemmas = preprocessor.lemmatize_text(test_sentence)
    print(f"Original: '{test_sentence}'")
    print(f"Lemmas: {lemmas}")

    # 5. Test pos_tag_text
    print("\n--- POS Tagging ---")
    pos_tags = preprocessor.pos_tag_text(test_sentence)
    print(f"Original: '{test_sentence}'")
    pprint.pprint(pos_tags)


if __name__ == "__main__":
    main()
