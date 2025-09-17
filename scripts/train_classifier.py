# scripts/train_classifier.py
from pathlib import Path
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATA_DIR = Path("data/processesData")
TRAIN_FILE = str(DATA_DIR / "clauses_train.csv")
VAL_FILE = str(DATA_DIR / "clauses_val.csv")
OUTPUT_DIR = Path("outputs/legalbert_classifier")

# 1. load csv dataset
data_files = {"train": TRAIN_FILE, "validation": VAL_FILE}
ds = load_dataset("csv", data_files=data_files)

# 2. label mapping
unique_labels = sorted(set(ds["train"]["clause_label"]))
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}

def map_label(example):
    example["label"] = label2id[example["clause_label"]]
    return example

ds = ds.map(map_label)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(examples["paragraph_text"], truncation=True, max_length=512)

tokenized = ds.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    save_total_limit=2,
    logging_steps=50,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).astype(float).mean().item()
    # optionally compute f1/precision/recall using sklearn
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model()
