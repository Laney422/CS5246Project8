import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, Bidirectional, MaxPooling1D, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from nltk.stem import WordNetLemmatizer
import nltk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
nltk.download('wordnet')
from sklearn.metrics import classification_report, confusion_matrix
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import matplotlib.pyplot as plt
import seaborn as sns



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
    
    # Remove leading and trailing spaces
    sentence = sentence.strip()
    
    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def train_tokenizer(filepath, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    tokenizer.train(files=[filepath], trainer=trainer)
    return tokenizer


def load_data_bpe(filepath, tokenizer, maxlen):
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
        
    if filepath == 'dataset/w_scores/bert_suicide_prob.csv':
        class_type = ['suicide', 'non-suicide']
    elif filepath == 'dataset/w_scores/bert_all_prob.csv':
        class_type = ['adhd', 'anxiety', 'bipolar', 'depression', 'eatingdisorders', 'ocd', 'ptsd', 'normal_positive','selfharm', 'stress']
        
    # Filter out the classes
    df = df[df['class'].isin(class_type)]
    
    df['Processed_Text'] = df['text'].apply(preprocess_text)
    encoded_texts = [tokenizer.encode(text).ids for text in df['Processed_Text']]
    padded_sequences = pad_sequences(encoded_texts, maxlen=maxlen, padding='post', truncating='post')
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['class'])
    labels = to_categorical(labels)
    
    return padded_sequences, labels, df.index  # Return the DataFrame index to track filtered rows

def prepare_dataset(filepath, tokenizer, maxlen, has_label=True):
    padded_sequences, labels, indices = load_data_bpe(filepath, tokenizer, maxlen)
    
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    df_filtered = df.loc[indices]  # Ensure additional features are aligned with filtered rows
    
    additional_features_df = pd.get_dummies(df_filtered.drop(['text', 'id', 'class', 'predictions', 'token', 'token_count', 'sentiment', 'text_count'], axis=1, errors='ignore'))
    
    padded_sequences_df = pd.DataFrame(padded_sequences, index=indices)  # Use the same index to ensure alignment
    final_features = pd.concat([padded_sequences_df, additional_features_df], axis=1)
    
    return final_features, labels



def load_embeddings(filepath):
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def analyze_dataset(filepath):
    # Calculate basic statistics
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    text_lengths = df['text'].str.len()
    text_length_95th_percentile = np.percentile(text_lengths, 95)

    # Print statistics
    print(f"95th percentile of text lengths: {text_length_95th_percentile:.2f}")

    # Plot text length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, bins=30, kde=True)
    plt.axvline(text_length_95th_percentile, color='r', linestyle='--', label='95th percentile')
    plt.title(f'Distribution of Text Lengths for Dataset {filepath}')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    
def build_model(word_index, embedding_matrix, embedding_dim, maxlen, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],  
                    input_length=maxlen,
                    trainable=False))
    
    model.add(Dropout(0.25))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(128, dropout=0.25)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0015)))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0015)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def train_model(model, X_train, y_train, batch_size, epochs, patience, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])
    return history

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions_labels = np.argmax(predictions, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    print(classification_report(y_test_labels, predictions_labels))
    cm = confusion_matrix(y_test_labels, predictions_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def main():
    filepath = 'dataset/w_scores/bert_all_prob.csv'
    embedding_filepath = 'glove.42B.300d.txt'
    vocab_size = 30000
    embedding_dim = 300
    max_len = 300
    batch_size = 512
    epochs = 100
    patience = 3
    num_classes = 10
    '''
    if filepath == dataset/w_scores/all_probs.csv':
        num_classes = 10
    elif filepath == 'bert_suicide_prob.csv':
        num_classes = 2
'''
    # Tokenizer training
    tokenizer = train_tokenizer(filepath, vocab_size)
    final_features, labels = prepare_dataset(filepath, tokenizer, max_len, has_label=True)
    input_length = final_features.shape[1]
    
    # Correct assignment of X and y using final_features and labels
    X = final_features
    y = labels
    
    # After preparing the dataset
    print(f"Number of samples in features: {final_features.shape[0]}")
    print(f"Number of samples in labels: {labels.shape[0]}")

    if final_features.shape[0] != labels.shape[0]:
        raise ValueError("The number of samples in features and labels must be the same.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Load word embeddings
    embeddings_index = load_embeddings(embedding_filepath)
    embedding_matrix = np.zeros((len(tokenizer.get_vocab()) + 1, embedding_dim))
    for word, i in tokenizer.get_vocab().items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    analyze_dataset(filepath)
    
    model = build_model(tokenizer.get_vocab(), embedding_matrix, embedding_dim, input_length, num_classes)


    # Train the model
    history = train_model(model, X_train, y_train, batch_size, epochs, patience, X_test, y_test)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    print(f'dataset: {filepath} (text & features) has been successfully trained and evaluated.')

if __name__ == '__main__':
    main()
