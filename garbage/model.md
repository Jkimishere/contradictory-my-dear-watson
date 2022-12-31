# Explaining the model architecture 



>This model uses **"bert-base-multilingual-cased"** to encode the sentences.


* Merge two sentences together. (from dataset in './data/')
* Encode with "bert-base-multilingual-cased" tokenizer from huggingface
* Input the tokenized tensor into a LSTM model (not final, still experimenting) (*currently, I will remove the attention mask from the output tensor.*)
* Forward pass, output.



