## Working Guid
1. Create a venv: ``` python -m venv venv ``` , ``` venv\Scripts\activate ```
2. ```pip install -r requirements.txt```
3. Put the training data and ```python data/preprocess.py```
4. Train tokenizer: ```python tokenizer.py```
5. Train LLADA model: ```python train.py```
6. Eval_1: ``` python sample.py ```
7. Eval_2: ``` python eval.py ```
8. Launch App: ``` python app.py ```