from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import ast
import re
from nltk.stem import WordNetLemmatizer

model_path = './Model/fine_tuned_base_bert'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prepare_data_for_prediction(texts, tokenizer, max_length=128):
    encoding = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encoding


def predict_with_probabilities_and_ids(texts, ids, tokenizer, model, device):
    model.eval()  
    probabilities_list = []
    

    for text in tqdm(texts, desc="Predicting"):
        with torch.no_grad(): 
            inputs = prepare_data_for_prediction([text], tokenizer)  
            inputs = {k: v.to(device) for k, v in inputs.items()} 
            
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)  
            probabilities_list.append(probabilities.cpu().numpy()[0])  
    
    return ids, probabilities_list
    

def string_to_list(row):
    try:
        return ast.literal_eval(row)
    except ValueError:
        return []
 
def predict_and_evaluate(test_df, tokenizer, model, device):
    model.eval()
    all_predictions = []
    all_true_labels = []
    probabilities_list = []

    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Predicting and Evaluating"):
        text = row['Text']
        true_labels_vector = row['Label']
        true_label = np.argmax(true_labels_vector)  # Convert the one-hot encoded vector to a class index
        all_true_labels.append(true_label)

        with torch.no_grad():
            inputs = prepare_data_for_prediction([text], tokenizer)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
            probabilities_list.append(probabilities.cpu().numpy()[0])
            predicted_labels = np.argmax(probabilities.cpu().numpy()[0])
            all_predictions.append(predicted_labels)
            
    all_predictions = np.array(all_predictions)
    
    df_prediction = pd.DataFrame({'id': test_df['id'], 'prediction': all_predictions.tolist(), 'ground_truth': all_true_labels, 'probabilities': probabilities_list})
    df_prediction.to_csv('bert_prob.csv', index=False)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    print("F1 score: ", f1)

    
    return probabilities_list


def preprocess_text(sen, lemmatization=True):
    
    if not isinstance(sen, str):
        return ""
    
    # Lowercase the text
    sentence = sen.lower()
    
    # Remove consecutive repeated characters
    sentence = re.sub(r'(.)\1{3,}', r'\1', sentence)
  
    # Remove mentions
    sentence = re.sub(r"@\w+", '', sentence)
    
    # Remove hashtag symbols
    sentence = re.sub(r'#(\w+)', r'\1', sentence)

    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    # Remove non-ASCII characters
    sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
    
    # Remove extra spaces
    sentence = re.sub(' +', ' ', sentence)
    
    # Split into words
    words = sentence.split()
    
    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def main():
    df = pd.read_csv('./emotion/val.csv')

    texts = df['Text'].apply(preprocess_text).astype(str).tolist()
    ids = df['id'].tolist()

    df['Label'] = df['Label'].apply(string_to_list)

    # prob_list = predict_and_evaluate(df, tokenizer, model, device)

    ids, pred_prob = predict_with_probabilities_and_ids(texts, ids, tokenizer, model, device)
    df = pd.DataFrame({'id': ids, 'prediction': pred_prob})
    df.to_csv('bert_prob_list.csv', index=False)
    
if __name__ == "__main__":
    main()

