import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import compute_metrics

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
df = pd.read_csv("data/Question_Classification_Dataset.csv")

def prepare_data(label_col):
    df_sub = df[['Questions', label_col]].rename(columns={'Questions': 'text', label_col: 'label'})
    labels = df_sub['label'].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    df_sub['label'] = df_sub['label'].map(label2id)
    dataset = Dataset.from_pandas(df_sub)
    return dataset.train_test_split(test_size=0.2), label2id, id2label

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

def train_model(label_col, model_dir):
    dataset, label2id, id2label = prepare_data(label_col)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"✅ Model saved at {model_dir}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model("Category0", "models/model_cat0")
    train_model("Category1", "models/model_cat1")
    train_model("Category2", "models/model_cat2")
