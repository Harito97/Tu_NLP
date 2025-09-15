import spacy
from typing import List, Dict
from src.core.dataset_loaders import load_conllu_data

class NamedEntityRecognizer:
    """
    A class to perform Named Entity Recognition using spaCy.
    """

    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Loads a spaCy language model.

        Args:
            model_name: The name of the spaCy model to load (e.g., 'en_core_web_sm').
        """
        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model '{model_name}' not found. Attempting to download...")
            spacy.cli.download(model_name)
            self._nlp = spacy.load(model_name)

    def recognize_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Recognizes named entities in the given text.

        Args:
            text: The input text string.

        Returns:
            A list of dictionaries, where each dictionary represents an entity
            with its text, label, start, and end offsets.
        """
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities

    def load_and_prepare_conllu_data(self, file_path: str) -> List[List[Dict[str, str]]]:
        """
        Loads CoNLL-U data and prepares it for NER tasks.
        Each sentence is a list of tokens, and each token is a dictionary of features.

        Args:
            file_path: The path to the CoNLL-U file.

        Returns:
            A list of sentences, where each sentence is a list of token dictionaries.
            Each token dictionary contains 'text' and potentially 'ner_tag' (if available).
        """
        token_lists = load_conllu_data(file_path)
        prepared_data = []
        for token_list in token_lists:
            sentence_tokens = []
            for token in token_list:
                token_dict = {"text": token["form"]}
                # Assuming 'misc' field might contain NER tags in some CoNLL-U formats
                # Or 'feats' or 'deprel' could be used for other token-level classifications
                if "ner_tag" in token.get("misc", {}):
                    token_dict["ner_tag"] = token["misc"]["ner_tag"]
                # Add other relevant fields if necessary, e.g., 'upos', 'xpos', 'deprel'
                sentence_tokens.append(token_dict)
            prepared_data.append(sentence_tokens)
        return prepared_data
