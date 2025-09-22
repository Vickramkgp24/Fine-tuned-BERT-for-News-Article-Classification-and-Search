from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os, config

class AGNewsPreparer:
    def __init__(self, save_path=config.DATA_ROOT):
        self.dataset = load_dataset("ag_news")
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def prepare_data(self, train_frac=0.12, test_frac=0.10, val_frac=0.10, random_state=42):
        # Subset selection
        train_subset = self.dataset['train'].select(range(int(train_frac * len(self.dataset['train']))))
        test_set = self.dataset['test'].select(range(int(test_frac * len(self.dataset['test']))))

        # Convert train subset to list of dicts
        train_subset_list = train_subset.to_list()
        labels = [item['label'] for item in train_subset_list]

        # Stratified train/val split
        train_list, val_list = train_test_split(
            train_subset_list, test_size=val_frac, stratify=labels, random_state=random_state
        )

        # Convert to DataFrames
        train_df = pd.DataFrame(train_list)
        val_df = pd.DataFrame(val_list)
        test_df = test_set.to_pandas()

        return train_df, val_df, test_df

    def save_data(self, train_df, val_df, test_df):
        train_df.to_csv(os.path.join(self.save_path, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.save_path, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.save_path, "test.csv"), index=False)
        print(f"✅ Files saved to: {self.save_path}")