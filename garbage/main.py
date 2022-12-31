from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")


text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')





import pandas as pd

pd.read_csv('../')
print(encoded_input)