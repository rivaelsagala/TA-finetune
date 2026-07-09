import json
import random
from pathlib import Path

DATASET_PATH = Path("/workspace/data/dataset/raft_dataset_finalv2.jsonl")
OUTPUT_DIR = Path("/workspace/data/dataset/split")
TEST_SIZE = 0.1
SEED = 3407


def normalize_documents(documents):
    if documents is None:
        return []
    if isinstance(documents, str):
        return [documents]
    if isinstance(documents, list):
        normalized = []
        for item in documents:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(item.get("content") or item.get("text") or json.dumps(item, ensure_ascii=False))
            else:
                normalized.append(str(item))
        return normalized
    return [str(documents)]


def normalize_record(record):
    record = dict(record)
    if "documents" in record:
        record["documents"] = normalize_documents(record["documents"])
    return record


def load_records(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Format JSON tidak valid di baris {line_number}: {exc}") from exc
            records.append(normalize_record(record))
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    records = load_records(DATASET_PATH)
    if not records:
        raise RuntimeError("Dataset kosong, tidak ada data yang bisa diproses.")

    random.seed(SEED)
    shuffled = records[:]
    random.shuffle(shuffled)

    split_index = int(len(shuffled) * (1 - TEST_SIZE))
    split_index = max(1, min(split_index, len(shuffled) - 1))

    train_records = shuffled[:split_index]
    val_records = shuffled[split_index:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    print(f"Dataset asli: {len(records)} baris")
    print(f"Train set: {len(train_records)} baris")
    print(f"Validation set: {len(val_records)} baris")
    print(f"Train disimpan ke: {train_path}")
    print(f"Validation disimpan ke: {val_path}")


if __name__ == "__main__":
    main()
