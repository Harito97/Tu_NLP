import sys
import os

# Add the project root to the Python path to allow imports from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer

def main():
    """
    Main function to test the implemented tokenizers.
    """
    # Instantiate the tokenizers
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # Sentences to test, as per the lab instructions
    test_sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",
    ]

    print("--- Tokenizer Evaluation ---")

    for i, sentence in enumerate(test_sentences):
        print(f"\n--- Sentence {i+1} ---\nOriginal: '{sentence}'")

        # Test SimpleTokenizer
        simple_tokens = simple_tokenizer.tokenize(sentence)
        # Filter out empty strings that might result from splitting
        simple_tokens = [token for token in simple_tokens if token]
        print(f"SimpleTokenizer Output: {simple_tokens}")

        # Test RegexTokenizer
        regex_tokens = regex_tokenizer.tokenize(sentence)
        print(f"RegexTokenizer Output:  {regex_tokens}")

if __name__ == "__main__":
    main()
