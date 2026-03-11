from transformers import AutoTokenizer
from pathlib import Path
import torch
import random
from tqdm import tqdm
import pandas as pd
import json
import os

class PrepareData:
    def __init__(
        self,
        tokenizer: str = "GSAI-ML/LLaDA-8B-Instruct",
        data_dir: str = "/data/sknigam/fineweb_10bt/sample/10BT",
        max_seq_length: int = 4096,
        id_mask_token: int = 126336,
        output_dir: str = "data",
        chunks_per_file: int = 10000
    ):
        self.tokenizer_name = tokenizer
        self.data_dir = Path(data_dir)
        self.max_seq_length = max_seq_length
        self.id_mask_token = id_mask_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunks_per_file = chunks_per_file
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size

    def prepare_dataset(self, eps: float = 1e-3):
        parquet_files = sorted(self.data_dir.glob("*.parquet"))
        device = torch.device("cpu")

        processed = []
        current_chunk = []
        chunk_count = 0
        file_count = 0

        def process_chunk(chunk_tokens):
            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.int32, device=device)
            t = random.random()
            p_mask = (1.0 - eps) * t + eps
            mask = torch.rand(chunk_tensor.size(0), device=device) < p_mask
            noisy = chunk_tensor.clone()
            noisy[mask] = self.id_mask_token
            return {
                "t": t,
                "input_ids": chunk_tensor,
                "noisy_input_ids": noisy,
                "mask": mask
            }

        for pq in parquet_files:
            df = pd.read_parquet(pq, engine="fastparquet")
            for text in tqdm(df["text"], desc=pq.name):
                tokens = self.tokenizer(text)["input_ids"]
                current_chunk.extend(tokens)

                while len(current_chunk) >= self.max_seq_length:
                    L = random.randint(1, self.max_seq_length) if random.random() < 0.01 else self.max_seq_length
                    chunk_tokens = current_chunk[:L]
                    processed.append(process_chunk(chunk_tokens))
                    current_chunk = current_chunk[L:]
                    chunk_count += 1

                    if chunk_count % self.chunks_per_file == 0:
                        out = self.output_dir / f"processed_chunk_{file_count:06d}.pt"
                        torch.save(processed, out)
                        processed = []
                        file_count += 1

        if processed:
            out = self.output_dir / f"processed_chunk_{file_count:06d}.pt"
            torch.save(processed, out)

        print(f"Total chunks: {chunk_count}")

class PrepareDataSFT:
    def __init__(
        self,
        data,
        tokenizer_name: str = "GSAI-ML/LLaDA-8B-Instruct",
        max_seq_length: int = 4096,
        output_dir: str = "data"
    ):
        self.data = data
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        self.vocab_size = self.tokenizer.vocab_size

    def prepare_and_save(self, output_file: str):
        records = []

        for ex in tqdm(self.data):
            prompt = ex["prompt"]
            response = ex["response"]
            logits_data = ex["logits"]

            p_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length).input_ids[0]
            r_ids = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=self.max_seq_length).input_ids[0]

            ids = torch.cat([p_ids, r_ids], dim=0)
            if ids.size(0) > self.max_seq_length:
                continue

            logits = torch.full((ids.size(0), self.vocab_size), -100.0)
            L = p_ids.size(0)

            for i, entry in enumerate(logits_data[: r_ids.size(0)]):
                pos = L + i
                if "full_logits" in entry:
                    fl = torch.tensor(entry["full_logits"])
                    if fl.numel() == self.vocab_size:
                        logits[pos] = fl
                else:
                    for t in entry.get("top_5", []):
                        if t["token_id"] < self.vocab_size:
                            logits[pos, t["token_id"]] = t["logit"]

            records.append({
                "input_ids": ids,
                "prompt_length": L,
                "llama_logits": logits
            })

        torch.save(records, os.path.join(self.output_dir, output_file))

def compute_and_save_logits(dataset, tokenizer, model, device, output_path, top_k=5):
    all_records = []

    for ex in tqdm(dataset):
        prompt = ex["prompt"]
        response = ex["response"]

        enc = tokenizer(prompt, response, return_tensors="pt")
        ids = enc.input_ids.to(device)

        with torch.no_grad():
            logits = model(ids).logits[0]

        p_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        r_ids = tokenizer(response, return_tensors="pt").input_ids[0]

        entries = []
        for i in range(r_ids.size(0)):
            lv = logits[p_len + i]
            if top_k is None:
                entries.append({
                    "chosen_token_id": int(r_ids[i]),
                    "full_logits": lv.cpu().tolist()
                })
            else:
                top = torch.topk(lv, k=top_k)
                entries.append({
                    "chosen_token_id": int(r_ids[i]),
                    "top_5": [
                        {"token_id": int(t), "logit": float(v)}
                        for t, v in zip(top.indices, top.values)
                    ]
                })

        all_records.append({
            "prompt": prompt,
            "response": response,
            "logits": entries
        })

    with open(output_path, "w") as f:
        json.dump(all_records, f)

def hermes_dataset_prod(example):
    sys = []
    hum = []
    gpt = ""

    for t in example["conversations"]:
        r = t["from"]
        v = t["value"].strip()
        if r == "system":
            sys.append(v)
        elif r == "human":
            hum.append(v)
        elif r == "gpt":
            gpt = v

    return {
        "prompt": "\n".join(sys + hum),
        "response": gpt
    }

if __name__ == "__main__":
    from transformers import AutoModel
    from datasets import load_dataset

    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

    data_root = "/data/sknigam/fineweb_10bt/sample/10BT"
    output_root = "/data/sknigam/fineweb_10bt/processed_data"
    os.makedirs(output_root, exist_ok=True)

    preparer = PrepareData(
        tokenizer="GSAI-ML/LLaDA-8B-Instruct",
        data_dir=data_root,
        max_seq_length=4096,
        id_mask_token=126336,
        output_dir=output_root,
        chunks_per_file=10000
    )

    preparer.prepare_dataset(eps=1e-3)

    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)

    ds = load_dataset("NousResearch/Hermes-3-Dataset")
    ds_train = ds["train"].map(
        hermes_dataset_prod,
        remove_columns=["conversations"],
        batched=False
    )

    logits_json = os.path.join(output_root, "hermes_with_logits.json")
    compute_and_save_logits(
        ds_train,
        tokenizer,
        model,
        device=device,
        output_path=logits_json,
        top_k=5
    )

    with open(logits_json, "r") as f:
        records = json.load(f)

    sft_preparer = PrepareDataSFT(
        data=records,
        tokenizer_name=model_name,
        max_seq_length=4096,
        output_dir=output_root
    )

    sft_preparer.init_tokenizer()
    sft_preparer.prepare_and_save(output_file="processed_sft.pt")