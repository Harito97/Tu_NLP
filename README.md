# Natural Language Processing (NLP) Course

This repository contains the materials for the Natural Language Processing (NLP) course.

**Natural-language processing (NLP)** is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to fruitfully process large amounts of natural language data. NLP helps computers understand human language and allows machines to communicate with us.

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

- Python 3.x

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/Harito97/Tu_NLP.git
   ```

2. Install Python packages

   ```sh
   pip install -r requirements.txt
   ```

3. Download spaCy language model (for some labs)

   ```sh
   python -m spacy download en_core_web_sm
   ```

## Project Structure

- **/lab/**: Contains detailed instructions and theory for each lab.
- **/src/**: Core Python source code for NLP tasks (tokenizers, vectorizers, models).
- **/spark_labs/**: Scala/Spark projects for big data NLP labs.
- **/test/**: Test scripts for each lab, including demonstrations of advanced model training.
- **/data/**: Raw datasets used in the labs.
- **/assets/**: Configuration files for model training.
- **/corpus/**: Processed data files (`.spacy`) ready for training.
- **/results/**: Output directory for trained models.
- **/log/**: Log files from training and other processes.

## Course Labs

This section outlines the practical labs designed to complement the theoretical lectures. Each lab focuses on implementing core NLP concepts and techniques.

- **Lab 1: Text Tokenization**
- **Lab 2: Count Vectorization**
- **Lab 3: TF-IDF Vectorization**
- **Lab 4: Word Embeddings with Word2Vec**
- **Lab 5: Text Classification**
- **Lab 6: Building an NLP Pipeline**
- **Lab 7: Token Classification (POS Tagging & NER)**
- **Lab 8: Language Models (N-gram Models)**
- **Lab 9: Large Language Models (LLMs)**
- **Lab 10: LLM Fine-tuning**
- **Lab 11: Prompt Engineering**
- **Lab 12: LLM Evaluation & Ethics**
- **Lab 13: Advanced Preprocessing**
- **Lab 14: Text Clustering**
- **Lab 15: Machine Translation**
- **Lab 16: Advanced Text Similarity**
- **Lab 17: Introduction to Spark NLP** (New!)

---

## Big Data NLP Labs (Scala & Apache Spark)

This project also includes a set of labs designed to introduce Natural Language Processing on a big data scale using Scala and Apache Spark. These labs live in the `spark_labs/` directory and are structured as a standard `sbt` project.

The goal of these labs is to learn how to use production-grade, distributed computing tools like Spark MLlib, rather than implementing algorithms from scratch.

### How to Run

0. Install Java (JDK 8 or higher) and [sbt](https://www.scala-sbt.org/download.html) if you haven't already.

1. **Navigate to the Spark Labs directory:**

   ```sh
   cd spark_labs
   ```

2. **Run the application:**

   ```sh
   sbt run
   ```

   - The first time you run `sbt run`, it will download all necessary Spark and Scala dependencies, which may take a few minutes.
   - The application will process the `data/c4-train.00000-of-01024-30K.json.gz` dataset.
   - Performance metrics will be saved to `log/lab17_metrics.log`.
   - Processed data samples will be saved to `results/lab17_pipeline_output.txt`.
   - Keep an eye on the console for output and the Spark UI link (`http://localhost:4040`).

To get started, navigate to the `spark_labs` directory and use `sbt` to run the examples. Instructions for each lab are located in the `lab/` directory, eg: `lab17_spark_nlp_intro.md`, ...

---

## Advanced Model Training

This project includes end-to-end examples of training custom NLP models from scratch.

### 1. Custom Word2Vec Model (Lab 4 Bonus)

- **Goal:** Train a `Word2Vec` embedding model on the `UD_English-EWT` dataset.
- **How to Run:**

  ```sh
  python test/lab4_embedding_training_demo.py
  ```

- **Output:** A trained `gensim` model is saved to `results/word2vec_ewt.model`.

### 2. Custom POS Tagger Model (Lab 7)

- **Goal:** Train a custom Part-of-Speech tagger using `spaCy` on the `UD_English-EWT` dataset.
- **How to Run (2 steps):**

  1. **Prepare Data:** Convert the CoNLL-U data to `.spacy` format.

     ```sh
     python test/prepare_pos_data.py
     ```

  2. **Train Model:** Run the `spaCy` training pipeline.

     ```sh
     python -m spacy train assets/pos_config.cfg --output results/pos_model --paths.train corpus/pos_train.spacy --paths.dev corpus/pos_dev.spacy
     ```

- **Test the Model:**

  ```sh
  python test/use_trained_pos_model.py
  ```

### 3. Custom NER Model (CoNLL-2003)

- **Goal:** Train a custom Named Entity Recognition model using `spaCy` on the standard `CoNLL-2003` dataset.
- **How to Run (2 steps):**

  1. **Download & Prepare Data:** This script will automatically download the dataset from Hugging Face and convert it.

     ```sh
     python test/prepare_conll2003_data.py
     ```

  2. **Train Model:** Run the `spaCy` training pipeline.

     ```sh
     python -m spacy train assets/ner_conll2003_config.cfg --output results/ner_conll2003_model --paths.train corpus/ner_train.spacy --paths.dev corpus/ner_dev.spacy
     ```

- **Test the Model:**

  ```sh
  python test/use_trained_ner_model.py
  ```

## Running Labs

To run the basic test script for any lab (e.g., Lab 1), navigate to the project root and execute:

```sh
python test/lab1_test.py
```

## Learning Roadmap

This roadmap provides a structured path through key concepts, methods, and applications of NLP.

### 1. Foundations

- **Mathematics:** Calculus, Linear Algebra, Statistics, Probability
- **Programming:** Python basics, PyTorch, Keras

### 2. Core NLP

- **Text Preprocessing:** Tokenization, Stemming, Lemmatization
- **Feature Extraction:** Bag-of-Words, TF-IDF
- **Information Extraction:** Rule-based & statistical methods, Phrase & Relationship Extraction
- **Part-of-Speech (POS) Tagging**
- **Named Entity Recognition (NER)**
- **Tense Identification**

### 3. Representations & Similarity

- **Word Embeddings:** Word2Vec, GloVe
- **Semantic Representations:** Contextual embeddings (BERT-based)
- **Text Similarity:** Jaccard, Cosine, Semantic similarity

### 4. NLP Tasks & Applications

- **Text Classification:** Supervised & unsupervised methods
- **Text Clustering:** K-Means and beyond
- **Sentiment Analysis**
- **Text Summarization:** Extractive & abstractive
- **Machine Translation:** Flores, mBART
- **Chatbots:** AIML, Rasa
- **Text Generation:** Language modeling, sequence-to-sequence

### 5. Speech & Multimodal NLP

- **Speech-to-Text (STT):** DeepSpeech
- **Text-to-Speech (TTS):** Tacotron
- **Speech recognition and processing across languages**

### 6. Modern NLP & LLMs

- **Large Language Models (LLMs):** GPT, BERT, mBART
- **Prompt Engineering & Fine-tuning**
- **Multilingual & Cross-lingual NLP**
- **Applications in real-world domains**

## Resources

### Books

- [Linear Algebra by Gilbert Strang](https://math.mit.edu/~gs/linearalgebra/)
- [Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [Mastering NLP with Python](http://file.allitebooks.com/20160919/Mastering%20Natural%20Language%20Processing%20with%20Python.pdf)
- [Neural Networks and Deep Learning](http.neuralnetworksanddeeplearning.com/)
- [Artificial Intelligence: A Modern Approach](https://github.com/pemagrg1/AI_class2022/blob/main/book/Artificial-Intelligence-A-Modern-Approach-4th-Edition-1-compressed.pdf)

### Datasets

- [Google NLP Datasets](https://ai.google/tools/datasets/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)
- [NLP Datasets on GitHub](https://github.com/niderhoff/nlp-datasets)

### Courses

- [Intro to Machine Learning (Udacity)](https://www.udacity.com/course/intro-to-machine-learning--ud120)
- [Machine Learning (Coursera)](https://www.coursera.org/learn/machine-learning/home/welcome)
- [Neural Networks and Deep Learning (Coursera)](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
- [Generative AI with LLMs (Coursera)](https://www.coursera.org/learn/generative-ai-with-llms)

## Tools & Technologies

### Programming & Frameworks

- **Python:** [Tutorial](https://www.tutorialspoint.com/python/index.htm)
- **Keras:** [Tutorial](https://www.tutorialspoint.com/keras/index.htm)
- **PyTorch:** [Tutorials](https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)

### Web Development

- **Flask:** [Tutorial](https.www.tutorialspoint.com/flask/index.htm)
- **FastAPI:** [Tutorial](https://fastapi.tiangolo.com/tutorial/)
- **Django:** [Tutorial](https://docs.djangoproject.com/en/4.0/intro/tutorial01/)

### Databases

- **MySQL:** [Tutorial](https://www.w3schools.com/mySQl/default.asp)
- **PostgreSQL:** [Tutorial](https://www.postgresqltutorial.com/)
- **MongoDB:** [Tutorial](https://docs.mongodb.com/manual/tutorial/)

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [NLP Roadmap from pemagrg1](https://github.com/pemagrg1/Natural-Language-Processing-NLP-Roadmap)

