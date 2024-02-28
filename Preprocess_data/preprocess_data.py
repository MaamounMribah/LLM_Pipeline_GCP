import pickle
from transformers import GPT2Tokenizer
from datasets import load_dataset
import sys

def preprocess_data(dataset_name='ag_news', task='sst2', split='train[:10%]'):
    dataset = load_dataset(dataset_name, task, split=split)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    preprocessed_data_path = 'preprocessed_data.pkl'
    with open(preprocessed_data_path, 'wb') as f:
        pickle.dump(tokenized_datasets, f)

    print("Data preprocessing completed.")
    print("Results saved to preprocessed_data.pkl file.")
    return preprocessed_data_path

if __name__ == "__main__":
    preprocess_data('ag_news', 'default', 'train[:10%]')