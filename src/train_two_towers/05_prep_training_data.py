import json
import random

max_lines = 500
min_frequency = 0  # Set the minimum frequency for tokenization

def main():

    embeddings_file = f"data/processed/ms_marco_clean_embeddings_{max_lines}_lines_minfreq_{min_frequency}.json"

    with open(embeddings_file, "r") as f:
        embeddings_dataset = json.load(f)
    
    print(f"Loaded embeddings dataset from {embeddings_file}")

    training_data = []

    for row in embeddings_dataset:
        for relevant_passage in row["passages"]:
            
            # select a random row that's != the current
            # row and select a random passage from that row
            # to use as the irrelevant document
            while True:
                irrelevant_row = random.choice(embeddings_dataset)
                if irrelevant_row != row:
                    break
            
            irrelevant_passage = random.choice(irrelevant_row["passages"])

            training_data.append({
                "query": row["query"],
                "relevant_doc": relevant_passage,
                "irrelevant_doc": irrelevant_passage
            })

    training_data_file = f"data/processed/ms_marco_training_data_{max_lines}_lines_minfreq_{min_frequency}.json"
    with open(training_data_file, "w") as f:
        json.dump(training_data, f, indent=4)

if __name__ == "__main__":
    main()