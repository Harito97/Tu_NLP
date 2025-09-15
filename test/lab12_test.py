import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llms.llm_evaluator import LLMEvaluator

def main():
    """
    Main function to test the LLMEvaluator and discuss ethical considerations.
    """
    print("--- LLMEvaluator and Ethical Considerations ---")

    try:
        evaluator = LLMEvaluator()

        # Task 1: ROUGE Score Calculation
        print("\n--- ROUGE Score Evaluation (Summarization) ---")
        predictions_rouge = [
            "The cat sat on the mat.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        references_rouge = [
            "The cat was on the mat.",
            "A quick brown fox jumps over the lazy dog.",
        ]
        rouge_scores = evaluator.calculate_rouge(predictions_rouge, references_rouge)
        print("Generated Summaries:")
        pprint.pprint(predictions_rouge)
        print("Reference Summaries:")
        pprint.pprint(references_rouge)
        print("ROUGE Scores:")
        pprint.pprint(rouge_scores)

        # Task 2: BLEU Score Calculation
        print("\n--- BLEU Score Evaluation (Translation/Generation) ---")
        predictions_bleu = [
            "The cat is on the mat.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        # BLEU expects references as List[List[str]]
        references_bleu = [
            ["The cat is on the mat.", "A cat is on the mat."],
            ["The quick brown fox jumps over the lazy dog."],
        ]
        bleu_scores = evaluator.calculate_bleu(predictions_bleu, references_bleu)
        print("Generated Texts:")
        pprint.pprint(predictions_bleu)
        print("Reference Texts:")
        pprint.pprint(references_bleu)
        print("BLEU Scores:")
        pprint.pprint(bleu_scores)

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        print("Please ensure 'evaluate' library is installed.")

    # Task 3: Ethical Discussion (Conceptual)
    print("\n--- Ethical Discussion: LLMs and Bias ---")
    print("Scenario: An LLM is used to generate job descriptions for a tech company.")
    print("When prompted for 'software engineer', it consistently generates descriptions with male pronouns and mentions of 'bro-culture'.")
    print("When prompted for 'nurse', it consistently generates descriptions with female pronouns and mentions of 'caring' and 'empathy'.")
    print("\nDiscussion Points:")
    print("1. What kind of bias is evident here? (e.g., gender bias, occupational bias)")
    print("2. What are the potential negative impacts of such biased outputs?")
    print("3. How could this bias have originated in the LLM?")
    print("4. What mitigation strategies could be employed to reduce or eliminate this bias?")
    print("   - Data-centric approaches (e.g., debiasing training data, data augmentation)")
    print("   - Model-centric approaches (e.g., fine-tuning with debiased data, adversarial training)")
    print("   - Post-processing approaches (e.g., filtering outputs, rephrasing)")
    print("   - Human-in-the-loop approaches (e.g., human review, feedback mechanisms)")
    print("   - Transparency and user education (e.g., informing users about potential biases)")
    print("\nThis section is for conceptual discussion and does not involve code execution.")


if __name__ == "__main__":
    main()
