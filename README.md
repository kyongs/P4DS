# P4DS
FinQA-GPT-4o-Framework


### 1. Setting

```bash
$ conda create -n finqa python=3.12
$ conda activate finqa
$ pip install -r requirements.txt

$ source .env # make sure to set openai api key

```

### 2. How to Run Code

```bash
$ python3 finqa_inference.py
# result file : ./evaluate/predict.json
```

### 3. How to Evaluate

```bash
$ cd evaluate
$ python evalute.py predict.json test.json
# refer to finqa github (https://github.com/czyssrs/FinQA)
```