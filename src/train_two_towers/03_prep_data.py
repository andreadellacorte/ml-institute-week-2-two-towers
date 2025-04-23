from config import *

import json
import torch
from huggingface_hub import HfApi, HfFolder
from tqdm import tqdm

from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokenised_dataset

from train_word2vec.model import CBOW

model_path = f"./data/checkpoints/2025_04_22__12_00_00/cbow.{max_lines}lines.{embedding_dim}embeddings.{min_frequency}minfreq.5epochs.pth"
clean_dataset_tokenised_file = f"data/processed/ms_marco_clean_tokenised_{max_lines}_lines_minfreq_{min_frequency}.json"
clean_dataset_embeddings_file = f"./data/processed/ms_marco_clean_embeddings_{max_lines}_lines_minfreq_{min_frequency}.json"

def main():
    dFoo = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dFoo)
    vocab_to_int, _ = get_tokenised_dataset(clean_dataset, min_frequency)

    # create a tokenised verson of clean_dataset

    for row in clean_dataset:
        row["query"] = [vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in row["query"].split()]
        row["passages"] = [[vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in passage.split()] for passage in row["passages"]]

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
                passage_mean = torch.zeros(cbow.emb.weight.size(1))
                for i, id in enumerate(passage):
                    passage_mean += (cbow.emb.weight[id].detach() - passage_mean) / (i + 1)
                passage_mean = passage_mean.unsqueeze(0)
                processed_passages.append(passage_mean)
            return processed_passages

        for row in tqdm(clean_dataset, desc="Processing rows"):
            row['query'] = process_query(row["query"], cbow)
            row["passages"] = process_passages(row["passages"], cbow)
        
        # Convert tensors to lists for JSON serialization
        for row in clean_dataset:
            row['query'] = row['query'].squeeze(0).tolist()
            for i, passage in enumerate(row['passages']):
                row['passages'][i] = passage.squeeze(0).tolist()

        with open(clean_dataset_embeddings_file, "w") as f:
            json.dump(clean_dataset, f, indent=4)

        # upload the file to hugging face

        # Define the file paths and repository details
        file_paths = [
            clean_dataset_embeddings_file
        ]

        repo_id = "andreadellacorte/ml-institute-week-2-two-towers"  # Replace with your Hugging Face repo
        commit_message = "Upload checkpoint files"

        # Authenticate with Hugging Face
        api = HfApi()
        token = HfFolder.get_token()  # Ensure you have logged in using `huggingface-cli login`

        # Upload the files
        for file_path in file_paths:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,  # Keep relative path in repo
                repo_id=repo_id,
                repo_type="model",  # Change to "dataset" if uploading to a dataset repo
                commit_message=commit_message,
            )
if __name__ == "__main__":
    main()