import re

INPUT = "data/raw.txt"
OUTPUT = "data/processed.txt"

def clean_line(line):
    # remove non-printable characters
    line = re.sub(r"[^\x20-\x7E]", "", line)

    # normalize whitespace
    line = re.sub(r"\s+", " ", line)

    return line.strip()

cleaned = []

with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = clean_line(line)
        if len(line) > 0:
            cleaned.append(line)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for line in cleaned:
        f.write(line + "\n")

print(f"Processed {len(cleaned)} lines")
