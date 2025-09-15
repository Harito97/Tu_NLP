from transformers import pipeline
from typing import List, Dict

class LLMTasks:
    """
    A class to perform various tasks using pre-trained Large Language Models (LLMs)
    from the Hugging Face Transformers library.
    """

    def __init__(self):
        """
        Initializes the necessary Hugging Face pipelines.
        Models will be downloaded the first time they are used.
        """
        self._generator = None
        self._summarizer = None
        self._qa_pipeline = None

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """
        Generates text based on a given prompt using a pre-trained generative model.

        Args:
            prompt: The input text prompt.
            max_length: The maximum length of the generated text.

        Returns:
            The generated text string.
        """
        if self._generator is None:
            print("Loading text generation model (gpt2)... This may take a moment.")
            self._generator = pipeline("text-generation", model="gpt2")
        
        result = self._generator(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]['generated_text']

    def summarize_text(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        """
        Summarizes a given text using a pre-trained summarization model.

        Args:
            text: The input text to summarize.
            max_length: The maximum length of the summary.
            min_length: The minimum length of the summary.

        Returns:
            The summarized text string.
        """
        if self._summarizer is None:
            print("Loading summarization model (t5-small)... This may take a moment.")
            self._summarizer = pipeline("summarization", model="t5-small")
        
        result = self._summarizer(text, max_length=max_length, min_length=min_length)
        return result[0]['summary_text']

    def answer_question(self, question: str, context: str) -> Dict[str, str]:
        """
        Answers a question based on a provided context using a pre-trained Q&A model.

        Args:
            question: The question to answer.
            context: The context text from which to find the answer.

        Returns:
            A dictionary containing the answer and other details.
        """
        if self._qa_pipeline is None:
            print("Loading question answering model (distilbert-base-uncased-distilled-squad)... This may take a moment.")
            self._qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        
        result = self._qa_pipeline(question=question, context=context)
        return result
