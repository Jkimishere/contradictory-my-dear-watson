# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')



import pandas as pd

df = pd.read_csv('./data/train.csv')


data = df

data['text'] = df.apply(lambda x: '', axis=1)


for i, row in data.iterrows():
    data['text'][i] = data['premise'][i] + '. ' + df['hypothesis'][i]

data.drop(['premise'], axis=1, inplace=True)
data.drop(['hypothesis'], axis=1, inplace=True)
data.drop(['language'], axis=1, inplace=True)
data.drop(['lang_abv'], axis=1, inplace=True)



# a = model(data['text'][4], candidate_labels)['scores']
# print(candidate_labels[a.index(max(a))])

# premise = sequence
# hypothesis = f'This example is {label}.'

# run through model pre-trained on MNLI
x = tokenizer.encode('I am stupid', 'I have 0 iq', return_tensors='pt',
                     truncation=True)
logits = nli_model(x.to('cpu'))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:,1]

print(prob_label_is_true)