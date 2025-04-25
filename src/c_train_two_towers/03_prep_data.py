from config import *

import json
import torch
from tqdm import tqdm

from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokeniser_dictionaries
import utils.hf_utils as hf_utils

from model import CBOW

model_path = f"./data/checkpoints/cbow/2025_04_22__10_00_00/cbow.{max_lines}lines.{embedding_dim}embeddings.{min_frequency}minfreq.5epochs.pth"
clean_dataset_tokenised_file = f"data/processed/ms_marco_clean_tokenised_{max_lines}_lines_minfreq_{min_frequency}.json"
clean_dataset_embeddings_file = f"./data/processed/ms_marco_clean_embeddings_{max_lines}_lines_minfreq_{min_frequency}.json"

def main():
    dFoo = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dFoo)
    vocab_to_int, _ = get_tokeniser_dictionaries(clean_dataset, min_frequency)

    # create a tokenised verson of clean_dataset

    for row in tqdm(clean_dataset, desc="Tokenising clean dataset..."):
        row["query"] = [vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in row["query"].split()]
        row["passages"] = [[vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in passage.split()] for passage in row["passages"]]
        row["is_selected"] = row["is_selected"]  # Keep the is_selected field as is
        
    with open(clean_dataset_tokenised_file, "w") as f:
        json.dump(clean_dataset, f, indent=4)
        
    print(f"Tokenised dataset saved to {clean_dataset_tokenised_file}")

    print(f"Loading model from {model_path}...")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cbow = CBOW(len(vocab_to_int), embedding_dim)
    cbow.load_state_dict(torch.load(model_path))
    cbow.eval()
    
    cbow.to(dev)

    print(f"Model loaded from {model_path} to {dev}.")

    print(f"Processing dataset...")

    with torch.no_grad():
        # call split on row["query"] and convert each item to embeddings with model
        def process_query(query, cbow):
            query_embeddings = [cbow.emb.weight[id].detach() for id in query]
            return torch.mean(torch.stack(query_embeddings), dim=0).unsqueeze(0)

        def process_passages(passages, cbow):
            processed_passages = []
            for passage in passages:
                passage_embeddings = [cbow.emb.weight[id].detach() for id in passage]
                passage_mean = torch.mean(torch.stack(passage_embeddings), dim=0).unsqueeze(0)
                processed_passages.append(passage_mean)
            return processed_passages

        for row in tqdm(clean_dataset, desc="Processing rows"):
            row['query'] = process_query(row["query"], cbow)
            row["passages"] = process_passages(row["passages"], cbow)
            row["is_selected"] = row["is_selected"]
        
        # Convert tensors to lists for JSON serialization
        for row in clean_dataset:
            row['query'] = row['query'].squeeze(0).tolist()
            for i, passage in enumerate(row['passages']):
                row['passages'][i] = passage.squeeze(0).tolist()

        print(f"Saving processed embeddings to {clean_dataset_embeddings_file}")

        with open(clean_dataset_embeddings_file, "w") as f:
            json.dump(clean_dataset, f, indent=4)

        print(f"Processed dataset saved to {clean_dataset_embeddings_file}")

        # Upload the file to hugging face
        hf_utils.save([clean_dataset_embeddings_file], repo_id, commit_message=f"Upload {clean_dataset_embeddings_file}")
if __name__ == "__main__":
    main()