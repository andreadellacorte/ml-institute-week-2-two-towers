import torch

class MarcoTWOTOWERS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = torch.tensor(self.data[idx]['query'], dtype=torch.float32)
        relevant_passage = torch.tensor(self.data[idx]['relevant_doc'], dtype=torch.float32)
        irrelevant_passage = torch.tensor(self.data[idx]['irrelevant_doc'], dtype=torch.float32)
        return query, relevant_passage, irrelevant_passage