from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datasets import Dataset
import config


class Evaluator:
    def __init__(self, model_path="output/bert-base-uncased", csv_path=None, label_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Load dataset if given
        if csv_path:
            df = pd.read_csv(csv_path)
            self.dataset = Dataset.from_pandas(df)
            self.texts = list(self.dataset["text"])
            self.labels = list(self.dataset["label"])
        else:
            self.dataset, self.texts, self.labels = None, None, None

        # Label names
        self.label_names = label_names or ["World", "Sports", "Business", "Sci/Tech"]

    def _tokenize_batch(self, texts):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=config.TRUNCATION,
            max_length=config.MAX_SEQ_LENGTH,
        )

    def get_predictions(self, texts, batch_size=32):
        preds = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self._tokenize_batch(batch_texts).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                preds.extend(pred_labels)
        return preds

    def evaluate(self):
        if self.texts is None or self.labels is None:
            raise ValueError("No dataset loaded. Please provide a CSV path during initialization.")

        y_pred = self.get_predictions(self.texts)
        y_true = self.labels

        print("🧪 Evaluation Metrics:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
        print(f"F1-score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
        print(f"F1-score (micro): {f1_score(y_true, y_pred, average='micro'):.4f}")

        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.label_names))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            cmap="Blues",
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig('heatmap.png', dpi=300)
        plt.show()
