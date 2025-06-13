import pandas as pd
from datasets import load_dataset


def load_bioasq_data():
    """Load BioASQ dataset and save as CSV."""
    print("Loading BioASQ dataset...")
    dataset = load_dataset("bioasq", "BioASQ-factoid")
    split = dataset['train']

    rows = []
    for idx, item in enumerate(split):
        rows.append({
            'idx': idx,
            'question': item.get('question', ''),
            'context': item.get('context', ''),
            'long_answer': item.get('exact_answer', ''),
            'final_decision': item.get('ideal_answer', ''),
            'labels': item.get('ideal_answer', '')
        })

    df = pd.DataFrame(rows)
    output_path = "./data/BIOASQ_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved BioASQ dataset to {output_path}")
    return df


if __name__ == "__main__":
    load_bioasq_data()
