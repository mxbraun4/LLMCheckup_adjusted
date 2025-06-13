import json
import pandas as pd
import os

def load_pubmedqa_fixed():
    """Load PubMed QA dataset with fixed label processing."""
    
    print("Loading PubMed QA dataset...")
    
    # Load the labeled data
    json_path = "./data/pubmedqa/data/ori_pqal.json"
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = []
    label_mapping = {'yes': 1, 'no': 0, 'maybe': 2}
    
    for key, item in data.items():
        # Use different possible fields for final decision
        final_decision = (
            item.get('final_decision') or 
            item.get('FINAL_DECISION') or 
            item.get('reasoning_required_pred') or
            'maybe'  # default fallback
        ).lower()
        
        # Map text labels to numeric
        numeric_label = label_mapping.get(final_decision, 2)  # default to 'maybe' (2)
        
        train_data.append({
            'idx': len(train_data),
            'question': item.get('QUESTION', ''),
            'context': ' '.join(item.get('CONTEXTS', [])),
            'long_answer': item.get('LONG_ANSWER', ''),
            'final_decision': final_decision,
            'labels': numeric_label,
            'reasoning_required': item.get('reasoning_required_pred', ''),
            'meshes': ', '.join(item.get('MESHES', [])),
            'year': item.get('YEAR', '')
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
    df = load_pubmedqa_fixed()
    print("\nFirst few examples:")
    print(df[['question', 'final_decision', 'labels']].head())
    print(f"\nLabel distribution:")
    print(df['labels'].value_counts())
    print(f"\nFinal decision distribution:")
    print(df['final_decision'].value_counts()) 