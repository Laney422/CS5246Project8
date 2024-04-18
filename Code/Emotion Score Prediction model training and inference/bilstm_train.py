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
import ast

class GoEmotionsDataset(Dataset):
    def __init__(self, data, tokenizer, num_classes=13):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = 128
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)
    
    def find_max_length(self):
        max_len = 0
        for idx in range(len(self.data)):
            text = self.data.iloc[idx, 1]  
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            max_len = max(max_len, len(tokens))

        return max_len
    
    def __getitem__(self, idx):
        text_entry = self.data.iloc[idx, 1]
        class_label = self.data.iloc[idx]['Label']
        id = self.data.iloc[idx, 0]
        if isinstance(text_entry, pd.Series):
            # If it's not, let's try converting the Series to a string (though this might indicate a larger issue)
            text = ' '.join(text_entry.astype(str))
        else:
            text = str(text_entry)
            
        # id = self.data.iloc[idx, 0]
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
        
        # Convert class_label to a tensor
        label_tensor = torch.tensor(class_label, dtype=torch.float)
        
        item = {
            'id': id,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label_tensor
        }
        
        return item







class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, n_layers, dropout, device):
        super(BiLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.target_size = 13
        self.embedding_dim = embedding_dim
        self.device = device

        
        
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            n_layers,
                            dropout=0.5,
                            bidirectional=True, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 13)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()
        
    
    def forward(self, input_ids, hidden, attention_mask=None):
        batch_size = input_ids.size(0)
        hidden = self.init_hidden(batch_size)
        embedded = self.embedding(input_ids)
        lstm_out, hidden = self.lstm(embedded, hidden)
        out = torch.sum(lstm_out,dim=1)
        out = torch.nn.functional.relu(out)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.device),
                    torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.device))
        return hidden 
    
    


class EmotionClassifier:
    def __init__(self, model, tokenizer, train_data, val_data, device, n_layers, output_size, dropout, max_len=128, batch_size=32, learning_rate=1e-3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.output_size = 13
        self.dropout = dropout
        self.train_dataloader = DataLoader(GoEmotionsDataset(train_data, tokenizer), batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(GoEmotionsDataset(val_data, tokenizer), batch_size=batch_size)
        
    def train(self, epochs):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for e in range(epochs):
            h = self.model.init_hidden(self.batch_size)
            counter = 0
            train_losses=[]
            
            for batch in self.train_dataloader:
                inputs, attention_mask, labels = (batch['input_ids'].to(self.device), 
                                batch['attention_mask'].to(self.device), 
                                batch['label'].to(self.device).float())

                counter += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                h = tuple([each.data for each in h])
                
                output, h = self.model(inputs, h)

                loss = criterion(output.squeeze(), labels.float())
                train_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if counter % 10 == 0:
                    print(f"Epoch {e+1}, Batch: {counter}, Loss: {loss.item()}")
                    
        torch.save(self.model.state_dict(), "model.pth")
                    
    def test(self):
        criterion = nn.CrossEntropyLoss()
        h = self.model.init_hidden(self.batch_size)
        
        with torch.no_grad():
            probs = []
            id = []
            count = 0
            total = 0
            loss = 0
            l = 0
            
            all_labels = []
            all_preds = []
            for batch in self.val_dataloader:
                inputs, attention_mask, labels = (batch['input_ids'].to(self.device), 
                                                batch['attention_mask'].to(self.device), 
                                                batch['label'].to(self.device).float())
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                batch_ids = batch['id']
                
                h = tuple([each.data for each in h])
                
                output, h = self.model(inputs, h)
                

                loss = criterion(output.squeeze(), labels.float())
                prob = F.softmax(output, dim=1)

                
                id.extend(batch_ids)
                probs.extend(prob.cpu().numpy())
  
                preds = torch.argmax(output, dim=1)
        
                for i, j in zip(preds, torch.argmax(labels, dim=1)):
                    all_preds.append(i.cpu().numpy())
                    all_labels.append(j.cpu().numpy())
                    if i == j:
                        count += 1
                total += labels.size(0)
                
                loss = criterion(output, labels.float())
                loss += loss
                l += 1
                
            acc = 100 * count / total
            test_loss = loss / l
            
            print(f"Test Loss: {test_loss}, Accuracy: {acc}")
            print(f"Test Accuracy: {acc}")
            
            predict_prob = pd.DataFrame({'Id': id, 'Predicted': probs})
            
            predict_prob.to_csv('predict_prob_rp.csv', index=False)
            print("Predictied probs saved to predict_prob_rp.csv")
            
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            
            f1 = f1_score(all_labels, all_preds, average='weighted')
            print(f"F1 Score: {f1}")

def data_preprocessing(df):
    stop_words = set(stopwords.words('english'))
    words_to_keeps = {'not, no'}
    stop_words = stop_words - words_to_keeps
    
    def clean_text(text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove non-word characters except "!"
        text = re.sub(r'[^\w\s!]', '', text)
        # Tokenize and remove stopwords
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    # Apply text cleaning
    df['Text'] = df['Text'].apply(clean_text)

    # Remove duplicated instances based on 'text' while keeping the first instance
    df = df.drop_duplicates(subset='Text', keep='first')

    # Identify and remove instances where one text has multiple different classes
    df_grouped = df.groupby('Text').nunique()
    duplicated_texts = df_grouped[df_grouped['Label'] > 1].index
    df = df[~df['Text'].isin(duplicated_texts)]

    return df

 


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('./Data/emotions_token_removePositive_transformed.csv')

    # Extracting the 'token' column
    tokens = df['Text']
    id = df['id']


    labels = df['Label']

    df = pd.DataFrame({'id': id, 'Text': tokens, 'Label': labels})


    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    train_df, val_df = train_test_split(df, test_size=0.2)
    
    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    hidden_dim = 128
    target_size = 5  
    n_layers = 2
    dropout = 0.3
    batch_size = 128
    learning_rate = 1e-3
    epochs = 10
    
    model = BiLSTM(embedding_dim, hidden_dim, vocab_size, target_size, n_layers, dropout, device).to(device)
    
    classifier = EmotionClassifier(model, tokenizer, train_df, val_df, device, n_layers, target_size, dropout, batch_size=batch_size, learning_rate=learning_rate)
    
    classifier.train(epochs)
    
    # Evaluate on validation set
    classifier.test()
    
    print("Training and evaluation complete. Predictions are saved to predictions.csv")

if __name__ == "__main__":
    main()


