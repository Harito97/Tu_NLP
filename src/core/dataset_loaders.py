import conllu
import os

def load_conllu_data(file_path):
    """
    Loads data from a CoNLL-U file.

    Args:
        file_path (str): The path to the CoNLL-U file.

    Returns:
        list: A list of TokenList objects, where each TokenList represents a sentence.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CoNLL-U file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    return conllu.parse(data)

def load_raw_text_data(file_path):
    """
    Loads raw text data from a file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    return data
