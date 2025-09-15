import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tasks.machine_translation import Translator
# Optional: Import LLMEvaluator for BLEU score calculation (requires 'evaluate' library)
try:
    from src.llms.llm_evaluator import LLMEvaluator
    has_evaluator = True
except ImportError:
    has_evaluator = False
    print("Warning: LLMEvaluator not available. BLEU score calculation will be skipped.")


def main():
    """
    Main function to test the Translator.
    """
    print("--- Translator Evaluation ---")
    print("NOTE: This script will download a pre-trained translation model the first time it runs.")
    print("This may take some time and data.")

    try:
        # 1. Instantiate Translator (English to French by default)
        translator = Translator(model_name="Helsinki-NLP/opus-mt-en-fr")

        # 2. Sample sentences
        english_sentences = [
            "Hello, how are you?",
            "Machine learning is a fascinating field.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        # Reference translations for BLEU score (if evaluator is available)
        # These should be human-quality translations
        reference_translations = [
            ["Bonjour, comment allez-vous ?"],
            ["L'apprentissage automatique est un domaine fascinant."],
            ["Le rapide renard brun saute par-dessus le chien paresseux."],
        ]

        # 3. Translate single sentence
        print("\n--- Single Sentence Translation ---")
        single_text = "Artificial intelligence is transforming the world."
        translated_single = translator.translate(single_text)
        print(f"Original: '{single_text}'")
        print(f"Translated: '{translated_single}'")

        # 4. Translate batch of sentences
        print("\n--- Batch Translation ---")
        translated_batch = translator.translate_batch(english_sentences)
        for i, original in enumerate(english_sentences):
            print(f"Original: '{original}'")
            print(f"Translated: '{translated_batch[i]}'")

        # 5. Calculate BLEU score (Bonus)
        if has_evaluator:
            print("\n--- BLEU Score Calculation (Bonus) ---")
            evaluator = LLMEvaluator()
            # Use the batch translated sentences as predictions
            bleu_scores = evaluator.calculate_bleu(translated_batch, reference_translations)
            print("BLEU Score:")
            pprint.pprint(bleu_scores)
        else:
            print("\nSkipping BLEU score calculation (LLMEvaluator not available).")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This might be due to missing dependencies or network issues while downloading models.")
        print("Please ensure 'transformers' is installed and you have an internet connection.")


if __name__ == "__main__":
    main()
