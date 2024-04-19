import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

nltk.download('wordnet')


def train_tokenizer(filepath, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    tokenizer.train(files=[filepath], trainer=trainer)
    return tokenizer

def load_data_bpe(filepath, tokenizer, maxlen):
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if filepath == 'dataset/text_aug/df_suicide_train_ag.csv' or 'dataset/w_scores/df_suicide_test.csv':
        class_type = ['suicide', 'non-suicide']
    elif filepath == 'dataset/text_aug/df_all_augmented.csv' or 'dataset/w_scores/df_test_augmented.csv':
        class_type = ['adhd', 'anxiety', 'bipolar', 'depression', 'eatingdisorders', 'ocd', 'ptsd', 'normal_positive','selfharm', 'stress']
        
    # Filter out the classes
    df = df[df['class'].isin(class_type)]
    encoded_texts = [tokenizer.encode(text).ids for text in df['cleaned_text']]
    padded_sequences = pad_sequences(encoded_texts, maxlen=maxlen, padding='post', truncating='post')
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['class'])
    return padded_sequences, to_categorical(labels)


def analyze_dataset(filepath):
    # Calculate basic statistics
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    text_lengths = df['cleaned_text'].str.len()
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
    

def load_embeddings(filepath):
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    return embeddings_index

def build_model(word_index, embedding_matrix, embedding_dim, maxlen, num_classes):
    model = Sequential([
        Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False),
        Dropout(0.25),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(128, dropout=0.25)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.0015)),
        Dropout(0.25),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0015))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, batch_size, epochs, patience, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    return model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    predictions_labels = np.argmax(predictions, axis=1)
    print(classification_report(y_test_labels, predictions_labels))
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test_labels, predictions_labels), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def main():
    train_filepath = 'dataset/text_aug/df_suicide_train_ag.csv'
    test_filepath = 'dataset/text_aug/df_suicide_test.csv'
    embedding_filepath = 'glove.42B.300d.txt'
    vocab_size = 30000
    embedding_dim = 300
    max_len = 500
    batch_size = 512
    epochs = 100
    patience = 3
    
    if train_filepath == 'dataset/text_aug/df_all_augmented.csv':
        num_classes = 10
    elif train_filepath == 'dataset/text_aug/df_suicide_train_ag.csv':
        num_classes = 2

    tokenizer = train_tokenizer(train_filepath, vocab_size)
    X_train, y_train = load_data_bpe(train_filepath, tokenizer, max_len)
    X_test, y_test = load_data_bpe(test_filepath, tokenizer, max_len)
    analyze_dataset(train_filepath)
    embeddings_index = load_embeddings(embedding_filepath)
    embedding_matrix = np.zeros((len(tokenizer.get_vocab()) + 1, embedding_dim))
    for word, i in tokenizer.get_vocab().items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    model = build_model(tokenizer.get_vocab(), embedding_matrix, embedding_dim, max_len, num_classes)
    history = train_model(model, X_train, y_train, batch_size, epochs, patience, X_test, y_test)
    evaluate_model(model, X_test, y_test)

    print('Training and evaluation for augmented dataset is completed.')

if __name__ == '__main__':
    main()
