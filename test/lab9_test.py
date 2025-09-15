import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llms.llm_tasks import LLMTasks

def main():
    """
    Main function to test the LLMTasks class.
    """
    print("--- LLMTasks Evaluation ---")
    print("NOTE: This script will download several large models from Hugging Face")
    print("the first time it runs. This may take a significant amount of time and data.")

    try:
        llm_tasks = LLMTasks()

        # Task 1: Text Generation
        print("\n--- Text Generation ---")
        prompt = "The future of AI is"
        generated_text = llm_tasks.generate_text(prompt, max_length=50)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")

        # Task 2: Text Summarization
        print("\n--- Text Summarization ---")
        long_text = """
        Large language models (LLMs) are a type of artificial intelligence (AI) program that can recognize and generate text, among other tasks. They are trained on vast datasets of text and code, allowing them to learn complex patterns and relationships within human language. This enables them to perform tasks like translation, summarization, question answering, and even creative writing. The development of LLMs has rapidly advanced in recent years, leading to powerful tools like GPT-3, LaMDA, and Llama. However, they also present challenges, including potential biases, the generation of misinformation, and significant computational resource requirements.
        """
        summarized_text = llm_tasks.summarize_text(long_text, max_length=50, min_length=20)
        print(f"Original Text (excerpt): '{long_text[:100]}...'")
        print(f"Summarized: '{summarized_text}'")

        # Task 3: Question Answering
        print("\n--- Question Answering ---")
        qa_context = """
        The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognisable structures in the world.
        """
        question = "Who designed the Eiffel Tower?"
        answer = llm_tasks.answer_question(question, qa_context)
        print(f"Context (excerpt): '{qa_context[:100]}...'")
        print(f"Question: '{question}'")
        print("Answer:")
        pprint.pprint(answer)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This might be due to missing dependencies or network issues while downloading models.")
        print("Please ensure 'transformers' is installed and you have an internet connection.")


if __name__ == "__main__":
    main()
