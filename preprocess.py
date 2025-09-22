import pandas as pd
import os, re, nltk
from nltk.corpus import stopwords
import config


class Preprocess:
    def __init__(self):
        self.data_path = config.DATA_ROOT
        os.makedirs(self.data_path, exist_ok=True)
        self.train_path =  config.TRAIN_CSV
        self.test_path = config.TEST_CSV
        self.val_path = config.VAL_CSV

    def load_csv(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        self.val_data = pd.read_csv(self.val_path)
        
    def load_nltk_componenets(self):
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
    
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]','',text)
        tokens = nltk.word_tokenize(text)
        tokens= [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, name=""):
        print(f"Preprocessing {name} dataset")
        df['clean_text'] = df['text'].apply(self.clean_text)
        #df.drop('text', axis=1, inplace=True)
        df.rename(columns={"text":"raw_text"}, inplace=True)
        df.rename(columns={"clean_text":"text"}, inplace=True)
        return df

    def call_for_preprocess(self):
        self.load_csv()
        self.load_nltk_componenets()
        train_df = self.preprocess_dataframe(self.train_data)
        val_df = self.preprocess_dataframe(self.val_data)
        test_df = self.preprocess_dataframe(self.test_data)
        return train_df, val_df, test_df
    
    def save_data(self, train_df, val_df, test_df):
        train_df.to_csv(self.train_path, index=False)
        val_df.to_csv(self.val_path, index=False)
        test_df.to_csv(self.test_path, index=False)
        print(f"✅ Files saved to: {self.data_path}")
