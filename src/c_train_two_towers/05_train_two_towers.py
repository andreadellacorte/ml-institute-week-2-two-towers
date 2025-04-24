from config import *

import torch
import json
import tqdm
import utils.hf_utils as hf_utils

import dataset
from doc_tower import DocTower
from query_tower import QueryTower

from utils.loss_utils import contrastive_loss

def main():
    traning_data_file = f"data/processed/ms_marco_training_data_{max_lines}_lines_minfreq_{min_frequency}.json"
    
    with open(traning_data_file, "r") as f:
        training_data = json.load(f)

    torch.manual_seed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dFoo = dataset.MarcoTWOTOWERS(training_data)
    dataloader = torch.utils.data.DataLoader(dFoo, batch_size=32, shuffle=True)
    
    docTower = DocTower(embedding_dim)
    queryTower = QueryTower(embedding_dim)

    docTower.to(dev)
    queryTower.to(dev)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(list(docTower.parameters()) + list(queryTower.parameters()), lr=learning_rate)

    # make a folder in data/checkpoints with the current date and time
    # to save the model in
    import os
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%M_%S")
    os.makedirs(f"data/checkpoints/{dt_string}", exist_ok=True)
    os.makedirs(f"data/checkpoints/{dt_string}/doc_tower", exist_ok=True)
    os.makedirs(f"data/checkpoints/{dt_string}/query_tower", exist_ok=True)

    epochs = 5
    for epoch in range(epochs):
        prgs = tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
        total_loss = 0.0

        for i, (query, relevant_doc, irrelevant_doc) in enumerate(prgs):

            query = query.to(dev)
            relevant_doc = relevant_doc.to(dev)
            irrelevant_doc = irrelevant_doc.to(dev)

            query = queryTower(query)
            pos = docTower(relevant_doc)
            neg = docTower(irrelevant_doc)
            
            loss = contrastive_loss(query, pos, neg, contrastive_loss_margin)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prgs.set_postfix(loss=total_loss / (prgs.n + 1))
    
        torch.save(docTower.state_dict(), f"data/checkpoints/{dt_string}/doc_tower/doc_tower_epoch_{epoch+1}.pth")
        torch.save(queryTower.state_dict(), f"data/checkpoints/{dt_string}/query_tower/query_tower_epoch_{epoch+1}.pth")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

        # Upload the file to hugging face
        hf_utils.save([clean_dataset_embeddings_file], repo_id, commit_message=f"Upload {clean_dataset_embeddings_file}")

if __name__ == "__main__":
    main()