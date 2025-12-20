from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"mask_token": "[MASK]"})

tokenizer.save_pretrained("./tokenizer")
