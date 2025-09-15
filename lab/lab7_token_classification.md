# Lab 7: Token Classification - Named Entity Recognition (NER)

## Setup

Before starting this lab, ensure you have installed all necessary project dependencies by running:

```bash
pip install -r requirements.txt
```

This lab uses the `spaCy` library, which requires a language model. Download the English small model (`en_core_web_sm`) by running:

```bash
python -m spacy download en_core_web_sm
```

## Objective

To understand and implement a fundamental token classification task: Named Entity Recognition (NER). You will learn to identify and classify named entities in text using a powerful NLP library.

## Theory

**Token Classification** is a type of NLP task where a label is assigned to each token (word) in a sequence. Examples include Part-of-Speech (POS) tagging, Chunking, and Named Entity Recognition.

**Named Entity Recognition (NER)** is the task of identifying and categorizing key information (entities) in text. These entities typically fall into predefined categories such as person names, organizations, locations, dates, etc.

**Example:**
"\textbf{Barack Obama} (PERSON) was born in \textbf{Honolulu} (LOC)."

**Approaches:**
*   **Rule-based:** Define patterns to match entities.
*   **Machine Learning-based:** Train models (e.g., Conditional Random Fields) on annotated data.
*   **Neural Network-based:** Modern approaches often use deep learning models (e.g., Bi-LSTMs, Transformers).

For this lab, we will leverage the capabilities of the `spaCy` library, which provides highly optimized and pre-trained NER models.

## Task 1: Setup `spaCy` Model

1.  **Install `spaCy` model:** If you haven't already, you need to download a `spaCy` language model. For English, `en_core_web_sm` is a good small model.
    ```bash
    python -m spacy download en_core_web_sm
    ```
    (Make sure `spaCy` itself is installed via `pip install -r requirements.txt` if you haven't done so).

## Task 2: `NamedEntityRecognizer` Implementation

1.  **Create the file:** `src/tasks/named_entity_recognition.py`.

2.  **Implement the `NamedEntityRecognizer` class:**
    *   The constructor `__init__(self, model_name: str = 'en_core_web_sm')` should load the `spaCy` language model.
    *   It should have an attribute `_nlp` to store the loaded `spaCy` pipeline.

3.  **Implement `recognize_entities(self, text: str) -> List[Dict[str, str]]`:**
    *   This method will take a raw text string.
    *   Process the text using the loaded `spaCy` model (`self._nlp(text)`).
    *   Iterate through the recognized entities (`doc.ents`).
    *   For each entity, extract its text, label, and start/end character offsets.
    *   Return a list of dictionaries, where each dictionary represents an entity (e.g., `{"text": "Barack Obama", "label": "PERSON", "start": 0, "end": 12}`).

## Task 3: Using UD_English-EWT for Token-level Data

The `UD_English-EWT` dataset provides rich token-level annotations in CoNLL-U format, which is excellent for various token classification tasks beyond just NER (e.g., Part-of-Speech tagging, dependency parsing).

1.  **Load CoNLL-U Data:**
    *   In your test file (`test/lab7_test.py` or a new script), import `NamedEntityRecognizer` and `load_conllu_data` from `src.core.dataset_loaders`.
    *   Load a CoNLL-U file from the dataset:
        ```python
        from src.tasks.named_entity_recognition import NamedEntityRecognizer
        from src.core.dataset_loaders import load_conllu_data

        # ... (your NER recognizer instantiation) ...

        conllu_file_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-dev.conllu"
        # Using the method added to NamedEntityRecognizer for convenience
        ner_recognizer = NamedEntityRecognizer() # Instantiate if not already
        prepared_data = ner_recognizer.load_and_prepare_conllu_data(conllu_file_path)

        print("\n--- Sample from UD_English-EWT CoNLL-U Data ---")
        if prepared_data:
            # Print first 3 sentences and their first few tokens
            for i, sentence in enumerate(prepared_data[:3]):
                print(f"Sentence {i+1}:")
                for j, token in enumerate(sentence[:10]): # Show first 10 tokens
                    print(f"  Token {j+1}: {token}")
                print("...")
        ```
    *   Inspect the `prepared_data` structure. Notice how each token has various fields (like `text`, `upos`, `feats`, `deprel`, etc.) which can be used for different token classification tasks.

2.  **Discussion:**
    *   How could you use this `prepared_data` to train a custom Part-of-Speech (POS) tagger or a dependency parser?
    *   What challenges might arise when converting these annotations to a format suitable for a specific machine learning model?

## Evaluation

*   Create a new test file: `test/lab7_test.py`.
*   Instantiate your `NamedEntityRecognizer`.
*   Test it with the following sentences and print the recognized entities:
    *   `"Apple Inc. was founded by Steve Jobs in Cupertino, California."`
    *   `"Dr. Smith works at Google in New York since January 1st, 2023."`

---

## Advanced Task: Training a Custom POS Tagger

While the tasks above focus on using a pre-trained model for NER, a more practical exercise with this dataset is to train a custom model for a task where we have ground-truth labels. The `UD_English-EWT` dataset is rich with Part-of-Speech (POS) tags, making it ideal for training a custom POS Tagger.

This project includes a demonstration of how to train a custom `spaCy` POS Tagger. The process is broken down into the following steps:

1.  **Data Preparation (`test/prepare_pos_data.py`):**
    *   A script is provided to read the `.conllu` files (`train` and `dev` sets).
    *   It extracts the Universal POS tag (UPOS) from the 4th column for each token.
    *   It converts this data into `spaCy`'s binary `.spacy` format, which is highly efficient for training. The output files are `corpus/pos_train.spacy` and `corpus/pos_dev.spacy`.

2.  **Configuration (`assets/pos_config.cfg`):**
    *   A `spaCy` configuration file will be created specifically for the `tagger` component. This file defines the model architecture, hyperparameters, and paths to the training/validation data.

3.  **Training:**
    *   Once the data and configuration are ready, the actual training is performed using `spaCy`'s command-line interface (CLI) via the `spacy train` command.

4.  **Evaluation and Use:**
    *   After training, the new custom POS tagger will be saved. This model can then be loaded to assign POS tags to new, unseen text.
