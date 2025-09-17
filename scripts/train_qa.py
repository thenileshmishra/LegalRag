# scripts/train_qa.py
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer
)
import numpy as np

# Config
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATA_DIR = Path("data/processesData")
TRAIN_FILE = DATA_DIR / "squad_train.json"
VAL_FILE = DATA_DIR / "squad_val.json"
OUTPUT_DIR = Path("outputs/qa_legalbert")

max_length = 384
doc_stride = 128

# 1. load and flatten SQuAD-like file into list of QA pairs
def load_squad_like(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    examples = []
    for doc in j["data"]:
        title = doc.get("title", "")
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "id": qa.get("id"),
                    "question": qa.get("question", ""),
                    "context": context,
                    "answers": qa.get("answers", [])
                })
    return examples

train_examples = load_squad_like(TRAIN_FILE)
val_examples = load_squad_like(VAL_FILE)

train_ds = Dataset.from_list(train_examples)
val_ds = Dataset.from_list(val_examples)
ds = DatasetDict({"train": train_ds, "validation": val_ds})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. tokenization / feature creation (standard HF SQuAD preprocessing)
def prepare_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # mapping from tokenized example to original
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # take first answer (CUAD dataset typically has one)
            answer = answers[0]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])

            sequence_ids = tokenized.sequence_ids(i)

            # find token start and end
            token_start_index = 0
            while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(sequence_ids) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # If answer outside the span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # find token indices
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

tokenized_ds = ds.map(
    prepare_features,
    batched=True,
    remove_columns=ds["train"].column_names
)

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

from transformers import IntervalStrategy

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    evaluation_strategy=IntervalStrategy.EPOCH,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-5,
    save_total_limit=2,
    logging_steps=100,
    fp16=False
)

# simple F1+EM compute (approx)
metric = evaluate.load("squad")

def compute_metrics(p):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
)

trainer.train()
trainer.save_model()
