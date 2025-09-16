# preprocess_cuad.py
import json
import csv
import os
from pathlib import Path

DATA_DIR = Path("data")  # adjust
CUAD_JSON = DATA_DIR / "CUAD_v1.json"
MASTER = DATA_DIR / "master_clauses.csv"
LABEL_GROUP = DATA_DIR / "label_group_xlsx"  # you can load mapping manually

OUT_DIR = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

# Load cuad json (SQuAD-like) 
with open(CUAD_JSON, "r", encoding="utf-8") as f:
    cuad = json.load(f)

squad = {"version": "v1", "data": []}
csv_rows = []
simplify_pairs = []

for doc in cuad:
    title = doc.get("title", "")
    paragraphs = doc["paragraphs"]  # depends on exactly how CUAD formatted; adapt keys
    for pid, p in enumerate(paragraphs):
        context = p["context"]
        qa_list = p.get("qas", [])
        # For SQuAD-like output
        if qa_list:
            squad_par = {"context": context, "qas": []}
            for qa in qa_list:
                qid = qa.get("id") or f"{title}_{pid}"
                question = qa.get("question", "Find clause")
                answers = qa.get("answers", [])
                # each answer should have 'text' and 'answer_start'
                squad_par["qas"].append({
                    "id": qid,
                    "question": question,
                    "answers": [{"text": a["text"], "answer_start": a["answer_start"]} for a in answers],
                    "is_impossible": False if answers else True
                })
                # CSV rows for clause classification
                if answers:
                    labels = question  # adjust mapping if question is clause label
                    csv_rows.append([title, pid, context.replace("\n", " "), labels])
                    # For simplification pairs: create a pseudo target using label + answer
                    # Ideally you have human-written simplified explanations; here we create placeholders.
                    for a in answers:
                        clause_text = a["text"]
                        source = f"CLAUSE_LABEL: {labels}\nCONTEXT: {context}\nCLAUSE: {clause_text}"
                        target = f"Explain this clause in plain English: {clause_text} (concise)"
                        simplify_pairs.append({"source": source, "target": target})
            squad["data"].append({"title": title, "paragraphs": [squad_par]})

# Write outputs
with open(OUT_DIR / "squad_train.json", "w", encoding="utf-8") as f:
    json.dump(squad, f, indent=2)

with open(OUT_DIR / "clauses_train.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["contract_id", "paragraph_id", "paragraph_text", "clause_label"])
    writer.writerows(csv_rows)

with open(OUT_DIR / "simplify_pairs.jsonl", "w", encoding="utf-8") as f:
    for pair in simplify_pairs:
        f.write(json.dumps(pair) + "\n")

print("Processed outputs in", OUT_DIR)
