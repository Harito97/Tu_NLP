import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ngram_language_model import NgramLanguageModel
from src.preprocessing.regex_tokenizer import RegexTokenizer

def main():
    """
    Main function to demonstrate training an NgramLanguageModel on the
    UD_English-EWT dataset.
    """
    print("--- Demonstrating N-gram Language Model Training (Lab 8) ---")

    # 1. Define the path to the dataset
    dataset_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-train.txt"

    # 2. Check if the dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # 3. Load a portion of the dataset to form a corpus
    print(f"\nLoading a sample corpus from: {dataset_path}")
    corpus = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    corpus.append(line)
                if len(corpus) >= 100: # Use first 100 lines for a quick training demo
                    break
        print(f"Corpus created with {len(corpus)} sentences.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # 4. Instantiate tokenizer and a bigram (n=2) language model
    tokenizer = RegexTokenizer()
    lm = NgramLanguageModel(tokenizer=tokenizer, n=2)

    # 5. Train the model on the corpus
    print("\nTraining the Bigram Language Model...")
    lm.fit(corpus)
    print("Training complete.")

    # 6. Demonstrate prediction
    print("\n--- Demonstrating Next Word Prediction ---")
    context = ['the']
    predictions = lm.predict_next_word(context)
    print(f"Top 5 most likely words to follow '{' '.join(context)}':")
    # Sort the predictions by probability in descending order and print the top 5
    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    pprint.pprint(sorted_predictions[:5])

    # 7. Demonstrate text generation
    print("\n--- Demonstrating Text Generation ---")
    seed_text = "the"
    generated_text = lm.generate_text(seed_text, length=15)
    print(f"Generated text from seed '{seed_text}':")
    print(f"-> '{generated_text}'")
    
    seed_text_2 = "i"
    generated_text_2 = lm.generate_text(seed_text_2, length=15)
    print(f"\nGenerated text from seed '{seed_text_2}':")
    print(f"-> '{generated_text_2}'")

if __name__ == "__main__":
    main()
