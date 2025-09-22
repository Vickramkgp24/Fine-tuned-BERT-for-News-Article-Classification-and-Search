import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import config

class HFTextClassifier:
    def __init__(self, train_ds, val_ds, test_ds, model_names, output_root=config.OUTPUT_ROOT):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_names = model_names
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def load_and_tokenize(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        def tokenize(batch):
            return tokenizer(batch['text'], padding='max_length', truncation=config.TRUNCATION, max_length=config.MAX_SEQ_LENGTH)
        return tokenizer, tokenize

    def fine_tune_model(self, model_name, num_labels=4, epochs=3, batch_size=8):
        tokenizer, tokenize = self.load_and_tokenize(model_name)

        train_dataset = self.train_ds.map(tokenize, batched=True)
        val_dataset = self.val_ds.map(tokenize, batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        output_dir = os.path.join(self.output_root, model_name.replace("/", "_"))
        os.makedirs(output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}
        )

        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Saved to: {output_dir}")

    def fine_tune_all(self):
        for model_name in self.model_names:
            self.fine_tune_model(model_name)
