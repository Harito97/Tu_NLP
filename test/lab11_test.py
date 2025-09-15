import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llms.prompt_engineer import PromptEngineer

def main():
    """
    Main function to test the PromptEngineer.
    """
    print("--- PromptEngineer Evaluation ---")
    print("NOTE: This script will download a pre-trained model (gpt2) the first time it runs.")
    print("This may take some time and data.")

    try:
        engineer = PromptEngineer(model_name="gpt2")

        # Task 1: Few-shot Text Classification
        print("\n--- Few-shot Text Classification ---")
        examples = [
            {"text": "This movie is fantastic!", "label": "Positive"},
            {"text": "I hate this film, it's terrible.", "label": "Negative"},
        ]
        text_to_classify = "The acting was superb, a truly great experience."
        classification = engineer.few_shot_classify(text_to_classify, examples)
        print(f"Text: '{text_to_classify}'")
        print(f"Predicted Sentiment: {classification}")

        # Task 2: Chain-of-Thought Reasoning
        print("\n--- Chain-of-Thought Reasoning ---")
        problem = "If a car travels at 60 miles per hour for 2 hours, how far does it travel?"
        reasoning_output = engineer.reason_with_cot(problem)
        print(f"Problem: '{problem}'")
        print(f"LLM Output (with CoT):\n{reasoning_output}")

        # Task 3: Role-playing/Persona Prompting
        print("\n--- Role-playing/Persona Prompting ---")
        role = "a grumpy old man"
        query = "Tell me about the weather today."
        role_play_output = engineer.role_play(role, query)
        print(f"Role: '{role}'")
        print(f"Query: '{query}'")
        print(f"LLM Output:\n{role_play_output}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This might be due to missing dependencies or network issues while downloading models.")
        print("Please ensure 'transformers' is installed and you have an internet connection.")


if __name__ == "__main__":
    main()
