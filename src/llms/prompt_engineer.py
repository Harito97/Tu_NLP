from transformers import pipeline
from typing import List, Dict

class PromptEngineer:
    """
    A class to demonstrate various prompt engineering techniques using an LLM.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initializes the PromptEngineer with a text generation pipeline.

        Args:
            model_name: The name of the generative model to use (e.g., 'gpt2').
        """
        print(f"Loading text generation model ({model_name})... This may take a moment.")
        self._generator = pipeline("text-generation", model=model_name)

    def few_shot_classify(self, text: str, examples: List[Dict[str, str]]) -> str:
        """
        Performs few-shot text classification.

        Args:
            text: The text to classify.
            examples: A list of dictionaries, each with 'text' and 'label' keys.

        Returns:
            The predicted label as a string.
        """
        prompt_parts = []
        for ex in examples:
            prompt_parts.append(f"Review: {ex['text']}\nSentiment: {ex['label']}")
        
        prompt_parts.append(f"Review: {text}\nSentiment:")
        full_prompt = "\n".join(prompt_parts)

        # Generate a short response to get the sentiment
        result = self._generator(full_prompt, max_new_tokens=5, num_return_sequences=1, pad_token_id=self._generator.tokenizer.eos_token_id)
        generated_text = result[0]['generated_text']
        
        # Extract the sentiment from the generated text
        # This is a simple heuristic and might need refinement for real-world use
        sentiment_line = generated_text.split("Sentiment:")[-1].strip()
        return sentiment_line.split("\n")[0].strip()

    def reason_with_cot(self, problem: str) -> str:
        """
        Encourages Chain-of-Thought reasoning for a problem.

        Args:
            problem: The problem statement.

        Returns:
            The LLM's step-by-step reasoning and answer.
        """
        prompt = f"Let's think step by step. {problem}\nReasoning:"
        result = self._generator(prompt, max_new_tokens=100, num_return_sequences=1, pad_token_id=self._generator.tokenizer.eos_token_id)
        return result[0]['generated_text']

    def role_play(self, role: str, query: str) -> str:
        """
        Instructs the LLM to adopt a specific role and respond to a query.

        Args:
            role: The role for the LLM to adopt (e.g., "a helpful assistant").
            query: The query to respond to.

        Returns:
            The LLM's response in the specified role.
        """
        prompt = f"You are {role}. {query}\nResponse:"
        result = self._generator(prompt, max_new_tokens=100, num_return_sequences=1, pad_token_id=self._generator.tokenizer.eos_token_id)
        return result[0]['generated_text']