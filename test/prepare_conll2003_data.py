import sys
import os
import spacy
from spacy.tokens import DocBin
from spacy.training.iob_utils import iob_to_biluo, biluo_tags_to_offsets
import logging
from tqdm import tqdm
import datasets

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_dataset_to_spacy(dataset_split, nlp):
    """Converts a Hugging Face dataset split (with CoNLL-2003 features) into a DocBin object."""
    doc_bin = DocBin()
    
    # Get the NER tag names (e.g., 'O', 'B-PER', 'I-PER', ...)
    ner_feature_names = dataset_split.features["ner_tags"].feature.names

    logging.info(f"Processing {len(dataset_split)} sentences...")
    for example in tqdm(dataset_split, desc="Converting examples"):
        words = example["tokens"]
        ner_tags_int = example["ner_tags"]
        
        # Convert integer tags to IOB string tags
        iob_tags = [ner_feature_names[i] for i in ner_tags_int]
        
        # Create a spaCy Doc
        doc = spacy.tokens.Doc(nlp.vocab, words=words)
        
        # Convert IOB to BILUO format and then to entity offsets
        biluo_tags = iob_to_biluo(iob_tags)
        try:
            # biluo_tags_to_offsets returns (start, end, label) tuples
            offsets = biluo_tags_to_offsets(doc, biluo_tags)
            
            # Create Span objects from the offsets
            spans = []
            for start, end, label in offsets:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    spans.append(span)
                else:
                    logging.warning(f"Could not create span from offsets ({start}, {end}) with label '{label}' for doc: {doc.text[:30]}...")
            
            doc.ents = spans
            doc_bin.add(doc)
        except ValueError as e:
            logging.warning(f"Could not set entities for sentence, skipping. Error: {e}. Tags: {biluo_tags}")

    return doc_bin

def main():
    """Main function to download CoNLL-2003 and convert it to .spacy format."""
    # Define paths
    output_dir = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/corpus"
    train_spacy_file = os.path.join(output_dir, "ner_train.spacy")
    dev_spacy_file = os.path.join(output_dir, "ner_dev.spacy")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load a blank English model for its vocab
    nlp = spacy.blank("en")

    # --- Download and process CoNLL-2003 dataset ---
    logging.info("Downloading CoNLL-2003 dataset from Hugging Face...")
    dataset = datasets.load_dataset("conll2003", trust_remote_code=True)
    logging.info("Download complete.")

    # --- Process Training Data ---
    train_doc_bin = convert_dataset_to_spacy(dataset["train"], nlp)
    train_doc_bin.to_disk(train_spacy_file)
    logging.info(f"Successfully created NER training data: {train_spacy_file}")

    # --- Process Development Data (using the validation split) ---
    dev_doc_bin = convert_dataset_to_spacy(dataset["validation"], nlp)
    dev_doc_bin.to_disk(dev_spacy_file)
    logging.info(f"Successfully created NER development data: {dev_spacy_file}")

if __name__ == "__main__":
    main()
