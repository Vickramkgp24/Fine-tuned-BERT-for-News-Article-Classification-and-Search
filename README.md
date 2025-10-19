# 🧠 Text Classification and Semantic Search Pipeline

This project implements a **complete end-to-end NLP pipeline** for **text classification** and **semantic search** using a **pre-trained HuggingFace Transformers model** such as `BERT`.  
The pipeline covers dataset preparation, text preprocessing, model fine-tuning, hyperparameter tuning, evaluation, and a semantic search utility.  
It is based on the **AG News** dataset.

---

## ⚙️ Project Structure

| File | Description |
|------|--------------|
| **config.py** | Central configuration for paths, model names, and hyperparameters. |
| **dataset.py** | Downloads and splits the AG News dataset into train, validation, and test sets. |
| **preprocess.py** | Implements the `Preprocess` class for cleaning, tokenizing, and lemmatizing text data. |
| **obtaining_representation.py** | Defines the `HFTextClassifier` class for fine-tuning models using the HuggingFace Trainer API. |
| **tuning.py** | Implements `ManualTuner` for manual grid search over learning rates and batch sizes. |
| **evaluator.py** | Provides the `Evaluator` class for model evaluation (accuracy, F1-score, confusion matrix). |
| **semantic_searcher.py** | Implements `SemanticSearcher` using FAISS for semantic similarity search on embeddings. |
| **train.py** | Main orchestrator script that runs all steps of the pipeline. |

---

## 🚀 Getting Started

### 🧩 Prerequisites

You’ll need **Python 3.8+** and the following libraries:

pip install transformers datasets pandas scikit-learn seaborn matplotlib torch faiss-cpu nltk

---

### 📦 NLTK Setup

The preprocessing module (`preprocess.py`) automatically downloads required NLTK resources,  
but you can also download them manually:

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

---

### ▶️ Running the Pipeline

Execute the main script to train and evaluate the model:

python train.py

This will automatically run:

1. Data preparation  
2. Preprocessing  
3. Model fine-tuning  
4. Manual hyperparameter tuning  
5. Evaluation  
6. Semantic search setup

---

## 📋 Pipeline Workflow

### 🧾 1. Data Preparation (`dataset.py`)

- Downloads the AG News dataset from HuggingFace Datasets.  
- Creates subsets and saves them as:

./data/train.csv
./data/val.csv
./data/test.csv

---

### 🧹 2. Text Preprocessing (`preprocess.py`)

Performs cleaning and normalization:

- Converts text to lowercase  
- Removes special characters and punctuation  
- Removes stopwords  
- Applies lemmatization  

All cleaned text is saved in a new column named `text`. These processed files overwrite the originals.

**Example:**
from preprocess import Preprocess
Preprocess().clean_data("data/train.csv")

---

### 🤖 3. Model Fine-Tuning (`obtaining_representation.py`)

Uses HuggingFace’s **Trainer API** to fine-tune a pre-trained model (such as `bert-base-uncased`).

**Steps:**
- Tokenizes input text  
- Trains model for a defined number of epochs  
- Saves the best-performing model to `./output/`

**Example:**
from obtaining_representation import HFTextClassifier
model = HFTextClassifier(model_name="bert-base-uncased")
model.train(train_df, val_df)

---

### 🔧 4. Manual Hyperparameter Tuning (`tuning.py`)

Performs grid search over combinations of learning rates and batch sizes defined in `config.py`.  
Reports the **best configuration** and corresponding validation accuracy.

**Example parameters:**
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
BATCH_SIZES =


**Sample output:**
Best configuration: Learning Rate = 3e-5, Batch Size = 32
Validation Accuracy: 94.6%

---

### 🔍 5. Semantic Search (`semantic_searcher.py`)

Uses the fine-tuned model’s encoder to generate sentence embeddings for all training samples.  
Builds a **FAISS index** for fast nearest-neighbor retrieval.

**Interactive search example:**
Enter a news query: NASA launches new satellite


**Output:**
Top 5 similar articles:

NASA successfully launches new Mars probe

SpaceX sends payload to orbit

Scientists celebrate successful satellite launch

Rocket launch marks milestone for agency

Space industry sees record growth

---

### 📊 6. Evaluation (`evaluator.py`)

Evaluates model performance on the test dataset using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

Also generates and saves a **confusion matrix heatmap** named `heatmap.png`.

**Example output:**
Accuracy: 94.5%
Precision: 94.6%
Recall: 94.5%
F1-score: 94.5%
Confusion matrix saved as heatmap.png

---

## 🔧 Configuration (`config.py`)

All pipeline parameters are stored centrally in `config.py`.

| Parameter | Description |
|------------|-------------|
| `MODEL_NAMES` | List of HuggingFace models to fine-tune (e.g., `["bert-base-uncased"]`) |
| `EPOCHS` | Number of training epochs |
| `BATCH_SIZE` | Default batch size for fine-tuning |
| `LEARNING_RATES` | List of learning rates for manual tuning |
| `BATCH_SIZES` | List of batch sizes for manual tuning |
| `MAX_SEQ_LENGTH` | Maximum token sequence length |
| `DATA_ROOT` | Directory to store datasets (`./data`) |
| `OUTPUT_ROOT` | Directory to store trained models (`./output`) |

**Example:**
MODEL_NAMES = ["bert-base-uncased"]
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
BATCH_SIZES =
MAX_SEQ_LENGTH = 128
DATA_ROOT = "./data"
OUTPUT_ROOT = "./output"

---

## 📈 Example Outputs

### ✅ Classification Results
| Metric | Value |
|---------|-------|
| Accuracy | 94.5% |
| Precision | 94.6% |
| Recall | 94.5% |
| F1-score | 94.5% |

---

### 🔍 Semantic Search Example

**Query:**
Apple releases new iPhone

**Results:**
Apple unveils iPhone 14 with improved camera

Tech industry reacts to Apple's new launch

Smartphone sales surge after Apple's release

New iOS update accompanies hardware refresh

Competitors respond to Apple's innovation


## 🧹 Cleaning Up

To delete all generated data and outputs:

rm -rf ./data ./output heatmap.png


## 🧠 Summary

This project delivers a modular and extendable **NLP pipeline** integrating:

- Data preprocessing  
- Transformer-based fine-tuning  
- Manual hyperparameter tuning  
- Model evaluation with detailed metrics  
- Semantic search using FAISS  

It can be adapted for any **text classification** or **semantic similarity** task with minimal code modification.

---
