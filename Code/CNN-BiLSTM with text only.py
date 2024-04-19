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
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import nltk
nltk.download('wordnet')

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
    
    # Remove leading and trailing spaces
    sentence = sentence.strip()
    
    # Split into words
    words = sentence.split()
    
    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def load_data_bpe(filepath, vocab_size, maxlen):
    df = pd.read_csv(filepath, engine='python')
    
    if filepath == 'dataset/text_only/df_suicide.csv':
        class_type = ['suicide', 'non-suicide']
    elif filepath == 'dataset/text_only/df_all.csv':
        class_type = ['adhd', 'anxiety', 'bipolar', 'depression', 'eatingdisorders', 'ocd', 'ptsd', 'normal_positive','selfharm', 'stress']
        
    # Filter out the classes
    df = df[df['class'].isin(class_type)]
    
    df['Processed_Text'] = df['text'].apply(preprocess_text)
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Customize training
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    
    # Train the tokenizer
    tokenizer.train(files=[filepath], trainer=trainer)
    
    # Encode the texts
    encoded_texts = [tokenizer.encode(text).ids for text in df['Processed_Text']]
    padded_sequences = pad_sequences(encoded_texts, maxlen=maxlen, padding='post', truncating='post')
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['class'])
    labels = to_categorical(labels)
    
    return padded_sequences, labels, tokenizer, label_encoder


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
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(32, dropout=0.25)))
    model.add(Dense(20, activation='relu', kernel_regularizer=l2(0.0015)))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0015)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Training the Model
def train_model(model, X_train, y_train, batch_size, epochs, patience):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])
    return history


# Evaluating the Model
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
    # File paths
    filepath = 'dataset/text_only/df_suicide.csv'
    embedding_filepath = 'glove.42B.300d.txt'
    

    # Parameters settings
    vocab_size = 30000
    embedding_dim = 300
    max_len = 300
    batch_size = 512
    epochs = 100
    patience = 3
    
    if filepath == 'dataset/text_only/df_suicide.csv':
        num_classes = 2
    elif filepath == 'dataset/text_only/df_all.csv':
        num_classes = 10
    
    
    # Load and process data
    X, y, tokenizer, _ = load_data_bpe(filepath, vocab_size, max_len)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    df = pd.read_csv(filepath, engine='python')


    # Prepare embedding matrix
    embeddings_index = load_embeddings(embedding_filepath)
    embedding_matrix = np.zeros((len(tokenizer.get_vocab()) + 1, embedding_dim))
    print(embedding_matrix.shape)
    for word, i in tokenizer.get_vocab().items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    analyze_dataset(filepath)
    
    # Build and train model
    model = build_model(tokenizer.get_vocab(), embedding_matrix, embedding_dim, max_len, num_classes)
    history = train_model(model, X_train, y_train, batch_size, epochs, patience)

    # Evaluate model

    evaluate_model(model, X_test, y_test)
    
    print(f'dataset: {filepath} (text only) has been successfully trained and evaluated.')



if __name__ == '__main__':
    main()
    
