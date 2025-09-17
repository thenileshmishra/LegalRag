# scripts/train_simplifier.py
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

MODEL_NAME = "google/flan-t5-base"
DATA_DIR = Path("data/processesData")
TRAIN_FILE = str(DATA_DIR / "simplify_pairs.jsonl")
VAL_FILE = str(DATA_DIR / "simplify_val.jsonl")
OUTPUT_DIR = Path("outputs/t5_simplifier")

# 1. load
ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE}, lines=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

max_input_length = 256
max_target_length = 128

def preprocess(batch):
    model_inputs = tokenizer(batch["source"], truncation=True, max_length=max_input_length, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target"], truncation=True, max_length=max_target_length, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    num_train_epochs=3,
    learning_rate=3e-5,
    save_total_limit=2,
    logging_steps=50
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model()
