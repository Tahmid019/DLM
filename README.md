## Download Dataset

[Dataset: Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

download Script:
```python
from huggingface_hub import snapshot_download
folder = snapshot_download(
                "HuggingFaceFW/fineweb", 
                repo_type="dataset",
                local_dir="./fineweb/",
                # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
                allow_patterns="sample/10BT/*")
```

## References

1. [Llada-from-Scratch](https://github.com/FredyRivera-dev/LLaDA-from-scratch)

## Working Guid
1. Create a venv: ``` python -m venv venv ``` , ``` venv\Scripts\activate ```
2. ```cd setup``` and ```./setup.sh```
3. Put the training data and ```pyt```
4. Train tokenizer: ```python tokenizer.py```
5. Train LLADA model: ```python train.py```
6. Eval_1: ``` python sample.py ```
7. Eval_2: ``` python eval.py ```
8. Launch App: ``` python app.py ```

