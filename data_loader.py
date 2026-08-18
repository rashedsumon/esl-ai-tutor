"""
data_loader.py: Downloads or generates synthetic ESL dialogue data for 
fine-tuning and populates ChromaDB with curriculum context.
"""
import os
import pandas as pd
from datasets import Dataset

DATA_PATH = "data/esl_dialogues.csv"

def download_or_generate_dataset() -> Dataset:
    """Auto-downloads or initializes local ESL learning dataset."""
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print("Generating ESL training dataset...")
        data = [
            {
                "instruction": "Guide an A2-level student who orders food in a café.",
                "input": "I want burger and water.",
                "output": "Great effort! A more polite way to say it is: 'I would like a burger and water, please.' What drink would you like?"
            },
            {
                "instruction": "Guide a B1-level student describing past weekend activities.",
                "input": "Last weekend I go to the beach and swimmed.",
                "output": "Nice! Remember past tense: 'Last weekend I went to the beach and swam.' Did you go with friends?"
            },
            {
                "instruction": "Provide scaffolding for an A1-level student introducing themselves.",
                "input": "My name John, I live Tokyo.",
                "output": "Hello John! Let's complete the verbs: 'My name IS John, and I LIVE IN Tokyo.' Where are you from?"
            }
        ]
        df = pd.DataFrame(data)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    return Dataset.from_pandas(df)

if __name__ == "__main__":
    ds = download_or_generate_dataset()
    print(f"Dataset ready with {len(ds)} samples.")