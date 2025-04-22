import torch
import json

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
    traning_data_file = f"data/processed/ms_marco_clean_training_data_{max_lines}_lines_minfreq_{min_frequency}.json"
    
    with open(traning_data_file, "r") as f:
        training_data = json.load(f)