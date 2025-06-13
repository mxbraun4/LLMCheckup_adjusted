import json
import pandas as pd
from datasets import load_dataset
import os

def load_pubmedqa_data():
    """Load PubMed QA dataset and convert to project format."""
    
    # Load using datasets library
    print("Loading PubMed QA dataset...")
    
    # Try to load the labeled version first
    try:
        dataset = load_dataset(
            "pubmed_qa",
            data_dir="./data/pubmedqa/data",
            name="pqa_labeled"
        )
        print("Successfully loaded pqa_labeled version")
    except Exception as e:
        print(f"Could not load pqa_labeled: {e}")
        # Fallback to manual loading
        return load_pubmedqa_manual()
    
    # Convert to DataFrame
    train_data = []
    
    # Process train split if available
    if 'train' in dataset:
        for item in dataset['train']:
            train_data.append({
                'idx': len(train_data),
                'question': item.get('question', ''),
                'context': ' '.join(item.get('context', {}).get('contexts', [])),
                'long_answer': item.get('long_answer', ''),
                'final_decision': item.get('final_decision', ''),
                'labels': item.get('final_decision', '')
            })
    
    # If no train split, use the full dataset
    else:
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                train_data.append({
                    'idx': len(train_data),
                    'question': item.get('question', ''),
                    'context': ' '.join(item.get('context', {}).get('contexts', [])),
                    'long_answer': item.get('long_answer', ''),
                    'final_decision': item.get('final_decision', ''),
                    'labels': item.get('final_decision', '')
                })
    
    df = pd.DataFrame(train_data)
    
    # Save as CSV
    output_path = "./data/PUBMEDQA_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved PubMed QA dataset to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def load_pubmedqa_manual():
    """Manual loading from JSON files if datasets library fails."""
    print("Attempting manual loading from JSON files...")
    
    # Load the original labeled data
    json_path = "./data/pubmedqa/data/ori_pqal.json"
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = []
    for key, item in data.items():
        train_data.append({
            'idx': len(train_data),
            'question': item.get('QUESTION', ''),
            'context': ' '.join(item.get('CONTEXTS', [])),
            'long_answer': item.get('LONG_ANSWER', ''),
            'final_decision': item.get('FINAL_DECISION', ''),
            'labels': item.get('FINAL_DECISION', '')
        })
    
    df = pd.DataFrame(train_data)
    
    # Save as CSV
    output_path = "./data/PUBMEDQA_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved PubMed QA dataset to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    df = load_pubmedqa_data()
    print("\nFirst few examples:")
    print(df.head())
    print(f"\nUnique labels: {df['labels'].value_counts()}") 