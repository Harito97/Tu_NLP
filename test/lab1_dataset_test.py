import sys
import os

# Add the project root to the Python path to allow imports from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.regex_tokenizer import RegexTokenizer


def main():
    """
    Main function to demonstrate using a tokenizer from 'src'
    on the actual UD_English-EWT dataset.
    """
    print("--- Demonstrating Tokenizer on UD_English-EWT Dataset ---")

    # 1. Define the path to the dataset
    # dataset_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-train.txt"
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "UD_English-EWT", "en_ewt-ud-train.txt"
    )

    # 2. Check if the dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # 3. Instantiate the tokenizer
    tokenizer = RegexTokenizer()

    # 4. Read and process the first few lines of the dataset
    print(f"\nReading first 5 lines from: {dataset_path}\n")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            line_count = 0
            for line in f:
                if line_count >= 5:  # Limit to the first 5 non-empty/non-comment lines
                    break

                line = line.strip()
                if not line or line.startswith("#"):  # Skip empty lines or comments
                    continue

                print(f"--- Original Line ---\n'{line}'")

                # Tokenize the line
                tokens = tokenizer.tokenize(line)

                print(f"Tokens: {tokens}\n")
                line_count += 1

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


if __name__ == "__main__":
    main()
