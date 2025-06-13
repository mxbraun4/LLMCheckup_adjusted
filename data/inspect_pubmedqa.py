import json
import pandas as pd

# Load and inspect the data
with open('./data/pubmedqa/data/ori_pqal.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Print first few examples to understand structure
print("Number of examples:", len(data))
print("\nFirst example structure:")
first_key = list(data.keys())[0]
print(f"Key: {first_key}")
example = data[first_key]
print("Available fields:", list(example.keys()))

for key, value in example.items():
    if isinstance(value, list):
        print(f"{key}: {len(value)} items - {value[:2] if value else 'empty'}")
    else:
        print(f"{key}: {value}")

print("\n" + "="*50)

# Check unique values for final decisions
final_decisions = []
for key, item in data.items():
    final_decisions.append(item.get('FINAL_DECISION', ''))

print("Unique final decisions:", set(final_decisions))
print("Final decision counts:")
for decision in set(final_decisions):
    count = final_decisions.count(decision)
    print(f"  {decision}: {count}")

# Let's also check if we have test data
with open('./data/pubmedqa/data/test_ground_truth.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"\nTest data size: {len(test_data)}")
test_example = list(test_data.values())[0]
print("Test example structure:", list(test_example.keys()))
print("Test final decision:", test_example.get('final_decision', 'N/A')) 