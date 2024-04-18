import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizer
import os
from bilstm_train import BiLSTM  

def preprocess_text(text, tokenizer):
    if pd.isna(text):
        return None, None
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Matches training setup
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoding['input_ids'], encoding['attention_mask']

def predict_emotion(text, model, tokenizer, device):
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    if input_ids is None or attention_mask is None:
        return "nan", "nan"  # Return "nan" for both outputs if input is NaN
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs[0]  # Assuming the logits are the first element of the tuple
        prob = F.softmax(logits, dim=1)
        prediction = torch.argmax(prob, dim=1)
    return prob.cpu().numpy().tolist(), prediction.item()




def check_for_nans(df, column_name='token'):
    nan_rows = df[df[column_name].isna()]
    if not nan_rows.empty:
        print(f"Found {len(nan_rows)} rows with NaNs in the '{column_name}' column.")
        print(nan_rows.head())
    else:
        print("No NaN values found in the '{column_name}' column.")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv('./Data/df_all.csv', encoding='utf-8', on_bad_lines='skip', lineterminator='\n')
    check_for_nans(df, 'token')  # Optional: you might remove this if NaN handling is now internal
    
    df = df.reset_index(drop=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_path = './Model/model.pth'
    
    model = BiLSTM(256, 128, tokenizer.vocab_size, 5, 2, 0.3, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    probs = []

    for text in tqdm(df['token'], desc="Predicting"):
        prob, pred = predict_emotion(text, model, tokenizer, device)
        predictions.append(pred)
        probs.append(prob)

    df_pred = pd.DataFrame({"id": df['id'], "predictions": predictions, "probs": probs})
    df_pred.to_csv('./output/all_bilstm_predictions.csv', index=False)
    print("File saved successfully.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
