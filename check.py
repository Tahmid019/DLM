import torch, glob, os

root = "/data/sknigam/fineweb_10bt/processed_data"
files = sorted(glob.glob(root + "/processed_chunk_*.pt"))

bad = []
for f in files:
    try:
        torch.load(f, map_location="cpu")
    except Exception as e:
        print("❌ BAD FILE:", f, "->", type(e).__name__)
        bad.append(f)

print("\nTotal files:", len(files))
print("Corrupted files:", len(bad))
