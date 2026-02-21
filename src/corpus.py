import json
import pickle
import zipfile
from pathlib import Path

ZIP_PATH = Path("data/corpus.zip")
OUTPUT_PATH = Path("data/processed_corpus.pkl")


def load_corpus_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        # pick a JSON/JSONL-like file inside the zip
        candidates = [
            n for n in z.namelist()
            if not n.endswith("/") and n.lower().endswith((".json", ".jsonl", ".ndjson"))
        ]
        if not candidates:
            raise FileNotFoundError("No .json/.jsonl/.ndjson file found inside corpus.zip")

        # Prefer corpus.json if present, else first candidate
        name = next((n for n in candidates if Path(n).name.lower() == "corpus.json"), candidates[0])

        with z.open(name, "r") as f:
            raw = f.read().decode("utf-8")

    raw = raw.lstrip()
    if raw.startswith("["):
        return json.loads(raw)  # JSON array
    else:
        # JSONL
        data = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data


def clean_text(text: str) -> str:
    text = (text or "").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def main():
    print("Loading corpus from zip...")
    protocols = load_corpus_from_zip(ZIP_PATH)
    print("Total protocols loaded:", len(protocols))

    processed = []
    for p in protocols:
        cleaned_text = clean_text(p.get("text", ""))

        processed.append({
            "protocol_id": p.get("protocol_id"),
            "source_file": p.get("source_file", ""),
            "title": p.get("title", ""),
            "icd_codes": p.get("icd_codes", []),
            "text": cleaned_text,
        })

    print("Saving processed corpus...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(processed, f)

    print("Done:", OUTPUT_PATH)


if __name__ == "__main__":
    main()