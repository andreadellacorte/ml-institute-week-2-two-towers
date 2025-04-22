import json
import torch
import sys
import os

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokenised_dataset
from train_word2vec import model as model_cbow

# Import the dataset and needed classes

dataset_id = "microsoft/ms_marco"  # Replace with your input file path
dataset_version = "v1.1"  # Replace with your desired dataset version
max_lines = None
min_frequency = 5  # Set the minimum frequency for tokenization
embedding_dim = 256

def main():
    dFoo = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dFoo)
    vocab_to_int, _ = get_tokenised_dataset(clean_dataset, min_frequency)

    # create a tokenised verson of clean_dataset

    for row in clean_dataset:
        row["query"] = [vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in row["query"].split()]
        row["passages"] = [[vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in passage.split()] for passage in row["passages"]]
    
    clean_dataset_tokenised_file = f"data/processed/ms_marco_clean_tokenised_{max_lines}_lines_minfreq_{min_frequency}.json"

    with open(clean_dataset_tokenised_file, "w") as f:
        json.dump(clean_dataset, f, indent=4)
        
    print(f"Tokenised dataset saved to {clean_dataset_tokenised_file}")

    # load the '2025_04_22__10_39_19.4.cbow.pth' model
    # use the model.CBOW file and also load the embeddings
    model_path = f"data/checkpoints/cbow.{max_lines}lines.{embedding_dim}embeddings.{min_frequency}minfreq.5.pth"

    print(f"Loading model from {model_path}...")
    
    cbow = model_cbow.CBOW(len(vocab_to_int), embedding_dim)
    cbow.load_state_dict(torch.load(model_path))
    cbow.eval()

    print(f"Model loaded from {model_path}")

    # call split on row["query"] and convert each item to embeddings with model
    for row in clean_dataset:
        for i, id in enumerate(row["query"]):
            row['query'][i] = cbow.emb.weight[id].detach()
        # average all the tensors in row['query'] into one
        row['query'] = torch.mean(torch.stack(row['query']), dim=0).unsqueeze(0)
        for i, passage in enumerate(row["passages"]):
            for j, id in enumerate(passage):
                passage[j] = cbow.emb.weight[id].detach()
            # average all the tensors in passage into one
            row["passages"][i] = torch.mean(torch.stack(passage), dim=0).unsqueeze(0)
    
    # Convert tensors to lists for JSON serialization
    for row in clean_dataset:
        row['query'] = row['query'].squeeze(0).tolist()
        for i, passage in enumerate(row['passages']):
            row['passages'][i] = passage.squeeze(0).tolist()

    clean_dataset_embeddings_file = f"data/processed/ms_marco_clean_embeddings_{max_lines}_lines_minfreq_{min_frequency}.json"

    with open(clean_dataset_embeddings_file, "w") as f:
        json.dump(clean_dataset, f, indent=4)

if __name__ == "__main__":
    main()