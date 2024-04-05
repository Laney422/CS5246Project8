import pandas as pd
from sklearn.model_selection import train_test_split
import os

            
def load_and_sample_dataset(file_path, sample=None, random_state=42):
    """
    Load a dataset and optionally sample it.
    
    Parameters:
    - file_path: Path to the dataset.
    - sample: Number of samples to draw from the dataset. If None, no sampling is performed.
    - random_state: Seed for the random number generator.
    
    Returns:
    - A DataFrame containing the loaded (and possibly sampled) dataset.
    """
    
    if file_path.endswith('dataset1.csv') or file_path.endswith('dataset2.csv'):
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    elif file_path.endswith('dataset3.csv'):
        df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
    


    if sample:
        df = df.sample(n=sample, random_state=random_state)
    return df

    




def split_and_save_dataset(df, output_prefix):
    """
    Split the DataFrame into train, test, and validation sets and save them.
    
    Parameters:
    - df: DataFrame to split.
    - output_prefix: Prefix for the output file names.
    """
    output_dir = os.path.dirname(output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    required_columns = ['text', 'class']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s) {', '.join(missing_columns)} in the DataFrame {output_prefix}")
    
    
    # Splitting dataset into train and temporary test set (70% train, 30% for split between test and validation)
    df_train, temp_test = train_test_split(df, train_size=0.7, stratify=df['class'], random_state=42)
    
    # Splitting the temporary test set into actual test and validation sets (50% each of the remaining 15% data)
    df_test, df_validation = train_test_split(temp_test, train_size=0.5, stratify=temp_test['class'], random_state=42)

    # Save the datasets
    df_train.to_csv(f'{output_prefix}_train.csv', index=False)
    df_test.to_csv(f'{output_prefix}_test.csv', index=False)
    df_validation.to_csv(f'{output_prefix}_validation.csv', index=False)
    
    print(f"Dataset split into train, test, and validation sets and saved with prefix {output_prefix}:")
    print(f"Train set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")
    print(f"Validation set size: {len(df_validation)}\n")
    
    
    
def main():
    # Define the specifics for each dataset
    datasets_info = {
        'dataset1': {'file_path': 'dataset/dataset1.csv', 'output_prefix': 'output/dataset1', 'sample_size': 4500},
        'dataset2': {'file_path': 'dataset/dataset2.csv', 'output_prefix': 'output/dataset2', 'sample_size': 4500},
        'dataset3': {'file_path': 'dataset/dataset3.csv', 'output_prefix': 'output/dataset3', 'sample_size': None},
    }

    # Process dataset1
    df1 = load_and_sample_dataset(datasets_info['dataset1']['file_path'], sample=datasets_info['dataset1']['sample_size'])
    df1_processed = process_dataset(df1, datasets_info['dataset1']['file_path'])
    split_and_save_dataset(df1_processed, datasets_info['dataset1']['output_prefix'])

    # Process dataset2
    df2 = load_and_sample_dataset(datasets_info['dataset2']['file_path'], sample=datasets_info['dataset2']['sample_size'])
    df2_processed = process_dataset(df2, datasets_info['dataset2']['file_path'])
    split_and_save_dataset(df2_processed, datasets_info['dataset2']['output_prefix'])

    # Process dataset3
    df3 = load_and_sample_dataset(datasets_info['dataset3']['file_path'], sample=datasets_info['dataset3']['sample_size'])
    df3_processed = process_dataset(df3, datasets_info['dataset3']['file_path'])
    split_and_save_dataset(df3_processed, datasets_info['dataset3']['output_prefix'])




if __name__ == '__main__':
    main()


