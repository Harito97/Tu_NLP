import sys
import os
import spacy
from spacy.tokens import DocBin

def main():
    """Loads a .spacy file and inspects the entities of the first few docs."""
    spacy_file = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/corpus/train.spacy"
    
    if not os.path.exists(spacy_file):
        print(f"Error: File not found at {spacy_file}")
        return

    print(f"--- Verifying entities in {spacy_file} ---")

    nlp = spacy.blank("en")
    db = DocBin().from_disk(spacy_file)
    docs = list(db.get_docs(nlp.vocab))

    print(f"Loaded {len(docs)} documents.")

    docs_with_ents = 0
    for i, doc in enumerate(docs[:10]): # Check the first 10 docs
        if doc.ents:
            print(f"\n--- Doc {i+1} ---")
            print(f"Text: {doc.text[:70]}...")
            print(f"Entities: {doc.ents}")
            docs_with_ents += 1
    
    print("\n---------------------")
    if docs_with_ents == 0:
        print("Verification FAILED: No entities found in the first 10 documents.")
        print("This confirms the data preparation script is likely bugged.")
    else:
        print(f"Verification PASSED: Found entities in {docs_with_ents}/10 documents.")

if __name__ == "__main__":
    main()

