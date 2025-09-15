import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.models.ngram_language_model import NgramLanguageModel

def main():
    """
    Main function to test the NgramLanguageModel.
    """
    print("--- NgramLanguageModel Evaluation ---")

    # 1. Define a small corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "The dog barks loudly.",
        "A quick fox is a clever fox.",
    ]
    print("\nCorpus:")
    pprint.pprint(corpus)

    # 2. Instantiate tokenizer and language model
    tokenizer = RegexTokenizer()
    # Let's test with a bigram model (n=2)
    lm = NgramLanguageModel(tokenizer=tokenizer, n=2)

    # 3. Train the model
    lm.fit(corpus)
    print("\nLanguage Model trained (Bigram, n=2).")

    # 4. Test predict_next_word
    print("\n--- Predict Next Word ---")
    contexts = [
        ["the"],
        ["fox"],
        ["dog"],
        ["quick", "brown"], # For n=3, this would be a context
    ]

    for context in contexts:
        print(f"Context: {context}")
        predictions = lm.predict_next_word(context)
        pprint.pprint(predictions)

    # 5. Test generate_text (Bonus)
    print("\n--- Generate Text ---")
    seed_text = "the quick"
    generated_text = lm.generate_text(seed_text, length=10)
    print(f"Seed: '{seed_text}'")
    print(f"Generated: '{generated_text}'")

    seed_text_long = "the dog"
    generated_text_long = lm.generate_text(seed_text_long, length=15)
    print(f"Seed: '{seed_text_long}'")
    print(f"Generated: '{generated_text_long}'")


if __name__ == "__main__":
    main()
