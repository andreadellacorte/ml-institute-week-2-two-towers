import torch

class QueryTower(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=embedding_dim, out_features=256)
        self.relu = torch.nn.ReLU()

    def forward(self, inpt):
        out = self.relu(self.fc1(inpt))
        return out