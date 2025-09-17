# preprocess_cuad.py
import json
import csv
import random
from pathlib import Path

DATA_DIR = Path("data/RawData")
CUAD_JSON = DATA_DIR / "CUAD_v1.json"

OUT_DIR = Path("data/processesData")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load CUAD JSON (SQuAD-like)
with open(CUAD_JSON, "r", encoding="utf-8") as f:
    cuad = json.load(f)

squad = {"version": "v1", "data": []}
csv_rows = []
simplify_pairs = []

for doc in cuad["data"]:
    title = doc.get("title", "")
    paragraphs = doc["paragraphs"]
    for pid, p in enumerate(paragraphs):
        context = p["context"]
        qa_list = p.get("qas", [])
        if qa_list:
            squad_par = {"context": context, "qas": []}
            for qa in qa_list:
                qid = qa.get("id") or f"{title}_{pid}"
                question = qa.get("question", "Find clause")
                answers = qa.get("answers", [])
                squad_par["qas"].append({
                    "id": qid,
                    "question": question,
                    "answers": [{"text": a["text"], "answer_start": a["answer_start"]} for a in answers],
                    "is_impossible": False if answers else True
                })
                if answers:
                    labels = question
                    csv_rows.append([title, pid, context.replace("\n", " "), labels])
                    for a in answers:
                        clause_text = a["text"]
                        source = f"CLAUSE_LABEL: {labels}\nCONTEXT: {context}\nCLAUSE: {clause_text}"
                        target = f"Explain this clause in plain English: {clause_text} (concise)"
                        simplify_pairs.append({"source": source, "target": target})
            squad["data"].append({"title": title, "paragraphs": [squad_par]})

# -------------------
# 1) Split SQuAD data
# -------------------
random.shuffle(squad["data"])
split_idx = int(0.9 * len(squad["data"]))
train_data = {"version": "v1", "data": squad["data"][:split_idx]}
val_data = {"version": "v1", "data": squad["data"][split_idx:]}

with open(OUT_DIR / "squad_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)
with open(OUT_DIR / "squad_val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2)

# ---------------------------
# 2) Split classification CSV
# ---------------------------
random.shuffle(csv_rows)
split_idx = int(0.9 * len(csv_rows))
train_rows = csv_rows[:split_idx]
val_rows = csv_rows[split_idx:]

with open(OUT_DIR / "clauses_train.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["contract_id", "paragraph_id", "paragraph_text", "clause_label"])
    writer.writerows(train_rows)

with open(OUT_DIR / "clauses_val.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["contract_id", "paragraph_id", "paragraph_text", "clause_label"])
    writer.writerows(val_rows)

# ---------------------------------
# 3) Split simplification JSON lines
# ---------------------------------
random.shuffle(simplify_pairs)
split_idx = int(0.9 * len(simplify_pairs))
train_pairs = simplify_pairs[:split_idx]
val_pairs = simplify_pairs[split_idx:]

with open(OUT_DIR / "simplify_pairs.jsonl", "w", encoding="utf-8") as f:
    for pair in train_pairs:
        f.write(json.dumps(pair) + "\n")

with open(OUT_DIR / "simplify_val.jsonl", "w", encoding="utf-8") as f:
    for pair in val_pairs:
        f.write(json.dumps(pair) + "\n")

print("Processed outputs in", OUT_DIR)
