from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device("cuda:0")



import pandas as pd

df = pd.read_csv('./data/test.csv')

import torch 
print(torch.cuda.is_available())

data = df


predarr = []
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
hypothesis = "Emmanuel Macron is the President of France"



for i, row in data.iterrows():
    input = tokenizer(data['premise'][i], data['hypothesis'][i], truncation=True, return_tensors="pt")
    model = model.to('cuda')
    output = model(input["input_ids"].to("cuda:0")) # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    print(prediction, label_names.index(max(prediction, key= prediction.get)))
    predarr.append(label_names.index(max(prediction, key= prediction.get)))
    if i % 100 == 0:
        print(i)

predfile = pd.read_csv('./data/sample_submission.csv')
predfile['prediction'] = predarr
predfile.to_csv('./preds.csv')
print(predarr)

