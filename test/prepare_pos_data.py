import sys
import os
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import logging
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.dataset_loaders import load_conllu_data

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_conllu_to_spacy_pos(conllu_path, nlp):
    """Converts CoNLL-U data into a DocBin object for POS tagging."""
    doc_bin = DocBin()
    conllu_sentences = load_conllu_data(conllu_path)
    
    logging.info(f"Processing {len(conllu_sentences)} sentences from {os.path.basename(conllu_path)}...")
    for sentence in tqdm(conllu_sentences, desc="Converting sentences"):
        words = [token['form'] for token in sentence]
        spaces = [True] * (len(words) - 1) + [False]
        tags = [token['upos'] for token in sentence]
        
        # Create a Doc object from the words and spaces
        doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
        
        # Create an Example object with the gold-standard POS tags
        example = Example.from_dict(doc, {"tags": tags})
        
        # Add the annotated doc to the DocBin
        doc_bin.add(example.reference)
            
    return doc_bin

def main():
    """Main function to convert CoNLL-U files to .spacy format for POS tagger training."""
    # Define paths
    train_conllu = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-train.conllu"
    dev_conllu = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-dev.conllu"
    output_dir = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/corpus"
    train_spacy_file = os.path.join(output_dir, "pos_train.spacy")
    dev_spacy_file = os.path.join(output_dir, "pos_dev.spacy")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load a blank English model for its vocab
    nlp = spacy.blank("en")

    # --- Process Training Data ---
    train_doc_bin = convert_conllu_to_spacy_pos(train_conllu, nlp)
    train_doc_bin.to_disk(train_spacy_file)
    logging.info(f"Successfully created POS training data: {train_spacy_file}")

    # --- Process Development Data ---
    dev_doc_bin = convert_conllu_to_spacy_pos(dev_conllu, nlp)
    dev_doc_bin.to_disk(dev_spacy_file)
    logging.info(f"Successfully created POS development data: {dev_spacy_file}")

if __name__ == "__main__":
    main()
