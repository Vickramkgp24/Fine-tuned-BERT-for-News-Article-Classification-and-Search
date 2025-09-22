import os
from dataset import AGNewsPreparer   # Dataset download and split
from preprocess import Preprocess       # Text pre-processing
from datasets import Dataset
from obtaining_representation import HFTextClassifier # HuggingFace Trainer wrapper
from tuning import ManualTuner      # Manual hyperparameter tuner
from semantic_searcher import SemanticSearcher
import config
from evaluator import Evaluator
import config

def prepare_data():
    preparer = AGNewsPreparer()
    traindf, valdf, testdf = preparer.prepare_data()
    preparer.save_data(traindf, valdf, testdf)
    return traindf, valdf, testdf

def preprocess_data():
    processor = Preprocess()
    traindf, valdf, testdf = processor.call_for_preprocess()
    processor.save_data(traindf, valdf, testdf)
    return traindf, valdf, testdf

def df_to_hf_dataset(df):
    return Dataset.from_pandas(df)

def main():
    
    output_dir = config.OUTPUT_ROOT
    modelnames = config.MODEL_NAMES
    num_labels = config.NUM_LABELS

    # Step 1: Data Preparation
    prepare_data()

    # Step 2: Preprocessing
    train_df, val_df, test_df = preprocess_data()
        
    # Step 3: Convert to HuggingFace datasets
    trainds = df_to_hf_dataset(train_df)
    valds = df_to_hf_dataset(val_df)
    testds = df_to_hf_dataset(test_df)

    # Step 4: HuggingFace Trainer (Automatic)
    hf_trainer = HFTextClassifier(trainds, valds, testds, modelnames, output_dir)
    hf_trainer.fine_tune_all()

    # Step 5: Manual Tuning
    lrs = config.LEARNING_RATES
    batch_sizes = config.BATCH_SIZES
    tuner = ManualTuner(trainds, valds, modelnames, output_dir, num_labels)
    results = tuner.tune_all(lrs, batch_sizes, epochs=config.EPOCHS)
    

    
    #Step 6 : Creating embeddings and printing top similarity text for give input
    searcher = SemanticSearcher()

    while True:
        user_query = input("\n🗞️ Enter your news-style query (or 'exit'): ")
        if user_query.lower() == "exit":
            break

        results = searcher.search(user_query)
        for res in results:
            print(f"\n🔹 Result #{res['rank']}")
            print(f"Category: {res['category']}")
            print(f"Title: {res['title']}...")
            print(f"Snippet: {res['snippet']}...")
            print(f"Similarity Score: {res['score']:.4f}")
        
    #Evalutaion
        
    evaluator = Evaluator(
        model_path="output/bert-base-uncased",
        csv_path=config.TEST_CSV,
        label_names=["World", "Sports", "Business", "Sci/Tech"],
    )
    evaluator.evaluate()
        
    
    
if __name__ == "__main__":
    main()
