import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score
import os
import config

class ManualTuner:
    def __init__(self, train_set, val_set, model_names, output_root = config.OUTPUT_ROOT , num_labels=4):
        """
        train_set, val_set: HuggingFace Datasets
        model_names: list of model names
        output_root: folder containing pre-trained/fine-tuned models
        num_labels: number of classification labels
        """
        self.train_set = train_set
        self.val_set = val_set
        self.model_names = model_names
        self.output_root = output_root
        self.num_labels = num_labels

    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_fn(batch):
            return tokenizer(batch['text'], padding='max_length', truncation=config.TRUNCATION, max_length=config.MAX_SEQ_LENGTH)

        return tokenizer, tokenize_fn

    def compute_metrics(self, eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}

    def manual_tuning(self, model_name, lrs, batch_sizes, epochs=2):
        tokenizer, tokenize_fn = self.load_tokenizer(model_name)

        train_encoded = self.train_set.map(tokenize_fn, batched=True)
        val_encoded = self.val_set.map(tokenize_fn, batched=True)

        cols = ['input_ids', 'attention_mask', 'label']
        train_encoded.set_format("torch", columns=cols)
        val_encoded.set_format("torch", columns=cols)

        best_acc = 0
        best_config = {}

        model_path = os.path.join(self.output_root, model_name.replace("/", "_"))

        for lr in lrs:
            for bs in batch_sizes:
                print(f"\n🚀 Training {model_name} | LR: {lr}, Batch Size: {bs}")

                args = TrainingArguments(
                    output_dir=f"tmp_output_{model_name.replace('/', '_')}",
                    eval_strategy="epoch",
                    save_strategy="no",
                    num_train_epochs=epochs,
                    per_device_train_batch_size=bs,
                    learning_rate=lr,
                    disable_tqdm=False,
                    report_to="none",
                    logging_steps=config.LOGGING_STEPS,
                    seed=config.SEED
                )

                model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels)

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_encoded,
                    eval_dataset=val_encoded,
                    tokenizer=tokenizer,
                    compute_metrics=self.compute_metrics
                )

                trainer.train()
                metrics = trainer.evaluate()
                acc = metrics["eval_accuracy"]
                print(f"📊 Validation Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_config = {"lr": lr, "batch_size": bs}

        print(f"\n✅ Best for {model_name}: LR = {best_config['lr']}, Batch Size = {best_config['batch_size']}, Accuracy = {best_acc:.4f}")
        return best_config, best_acc

    def tune_all(self, lrs, batch_sizes, epochs=2):
        results = {}
        for model_name in self.model_names:
            print(f"\n🔍 Manual tuning for: {model_name}")
            best_config, best_acc = self.manual_tuning(model_name, lrs, batch_sizes, epochs=epochs)
            results[model_name] = {"best_config": best_config, "best_acc": best_acc}
        return results
