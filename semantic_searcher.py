import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import pandas as pd
import config


class SemanticSearcher:
    def __init__(self, model_path="output/bert-base-uncased", csv_path=config.TRAIN_CSV):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer + base model (without classification head)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.base_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.base_model.eval()

        # Load and preprocess dataset
        train_set = pd.read_csv(csv_path)
        train_set = Dataset.from_pandas(train_set)

        self.train_texts = train_set["text"]
        self.train_raw_texts = train_set["raw_text"]
        self.train_labels = train_set["label"]

        # Build FAISS index
        print("⚙️ Generating training embeddings...")
        self.article_embeddings = self._generate_embeddings(self.train_texts)

        dim = self.article_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.article_embeddings)

        # Label map
        self.label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _generate_embeddings(self, texts, batch_size=16):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(self.device)
            with torch.no_grad():
                outputs = self.base_model(**inputs)
            pooled = self._mean_pooling(outputs, inputs["attention_mask"])
            pooled = pooled.cpu().numpy()
            embeddings.append(pooled)
        embeddings = np.vstack(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize
        return embeddings

    def search(self, query, k=5):
        # Encode query
        inputs = self.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            output = self.base_model(**inputs)
        query_embedding = self._mean_pooling(output, inputs["attention_mask"]).cpu().numpy()
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search top-k
        similarities, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            result = {
                "rank": i + 1,
                "category": self.label_map[self.train_labels[idx]],
                "title": self.train_raw_texts[idx][:60].strip().replace("\n", " "),
                "snippet": self.train_raw_texts[idx].strip(),
                "score": float(similarities[0][i]),
            }
            results.append(result)

        return results
