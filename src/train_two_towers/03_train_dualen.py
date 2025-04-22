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

# Load the dataset

class QueryTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=32)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=16)
        self.fc4 = torch.nn.Linear(in_features=16, out_features=1)
        self.relu = torch.nn.ReLU()

    def forward(self, inpt):
        x = self.relu(self.fc1(inpt))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

class DocTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=32)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=16)
        self.fc4 = torch.nn.Linear(in_features=16, out_features=1)
        self.relu = torch.nn.ReLU()

    def forward(self, inpt):
        x = self.relu(self.fc1(inpt))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

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
    model_path = "data/checkpoints/2025_04_22__10_39_19.5.cbow.pth"

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
    
    clean_dataset_embeddings_file = f"data/processed/ms_marco_clean_embeddings_{max_lines}_lines_minfreq_{min_frequency}.json"

    with open(clean_dataset_embeddings_file, "w") as f:
        json.dump(clean_dataset, f, indent=4)

if __name__ == "__main__":
    main()