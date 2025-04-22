import torch
import json
import tqdm
import dataset

dataset_id = "microsoft/ms_marco"  # Replace with your input file path
dataset_version = "v1.1"  # Replace with your desired dataset version
max_lines = 500
min_frequency = 0  # Set the minimum frequency for tokenization
embedding_dim = 256

# Load the dataset

class DocTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=embedding_dim, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=32)
        self.fc4 = torch.nn.Linear(in_features=32, out_features=16)
        self.relu = torch.nn.ReLU()

    def forward(self, inpt):
        x = self.relu(self.fc1(inpt))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

class QueryTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=embedding_dim, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=32)
        self.fc4 = torch.nn.Linear(in_features=32, out_features=16)
        self.relu = torch.nn.ReLU()

    def forward(self, inpt):
        x = self.relu(self.fc1(inpt))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

def contrastive_loss(query, relevant_doc, negative_doc, margin=0.2):
    """
    Contrastive loss function.
    Args:
        pos: Positive pair distance.
        neg: Negative pair distance.
        margin: Margin for the loss.
    Returns:
        Loss value.
    """

    # compuse pos and neg with cosine similarity
    cos_pos = torch.nn.functional.cosine_similarity(query, relevant_doc, dim=1)
    cos_neg = torch.nn.functional.cosine_similarity(query, negative_doc, dim=1)

    return torch.mean(torch.clamp(margin + cos_pos - cos_neg, min=0.0))

def main():
    traning_data_file = f"data/processed/ms_marco_training_data_{max_lines}_lines_minfreq_{min_frequency}.json"
    
    with open(traning_data_file, "r") as f:
        training_data = json.load(f)

    print(len(training_data))

    dFoo = dataset.MarcoTWOTOWERS(training_data)
    dataloader = torch.utils.data.DataLoader(dFoo, batch_size=1, shuffle=True)
    
    docTower = DocTower()
    queryTower = QueryTower()
    optimizer = torch.optim.Adam(list(docTower.parameters()) + list(queryTower.parameters()), lr=0.001)
    
    epochs = 5

    for epoch in range(epochs):
        prgs = tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
        total_loss = 0.0

        for i, (query, relevant_doc, irrelevant_doc) in enumerate(prgs):
            query = queryTower(query)
            pos = docTower(relevant_doc)
            neg = docTower(irrelevant_doc)
            
            loss = contrastive_loss(query, pos, neg, 0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prgs.set_postfix(loss=total_loss / (prgs.n + 1))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    main()