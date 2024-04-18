import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification, TrainingArguments, Trainer

from accelerate import DataLoaderConfiguration
import os
from torch.nn import BCEWithLogitsLoss
import ast




class GoEmotionsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Text']
        class_label = row['Label'] 
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        if isinstance(class_label, str):
            class_label = ast.literal_eval(class_label)
        
        label_tensor = torch.tensor(class_label, dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_tensor
        }



def main():
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    
    df1 = pd.read_csv('./Data/balanced_dataset.csv')

    tokens = df1['Text']
    ids = df1['id']

    labels = df1['Label']

    df = pd.DataFrame({'id': ids, 'Text': tokens, 'Label': labels})
    df = df.dropna(subset=['Text'])
    
    train_df, val_df = train_test_split(df, test_size=0.2)
    
    
    val_df.to_csv('./Data/val.csv', index=False)

    train_dataset = GoEmotionsDataset(train_df, tokenizer)
    val_dataset = GoEmotionsDataset(val_df, tokenizer)
    
    print(f"Number of training data points: {len(train_dataset)}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=13)
    model = model.to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        num_train_epochs=10,              # Total number of training epochs
        per_device_train_batch_size=8,   # Batch size per device during training
        per_device_eval_batch_size=8,    # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    model_path = './fine_tuned_base_bert'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
if __name__ == "__main__":
    main()