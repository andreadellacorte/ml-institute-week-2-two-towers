from config import *

import os
from datetime import datetime

import torch
import json
import tqdm
import utils.hf_utils as hf_utils

import c_train_two_towers.dataset as dataset
from c_train_two_towers.doc_tower import DocTower
from c_train_two_towers.query_tower import QueryTower

from utils.loss_utils import contrastive_loss
import wandb  # Add wandb import

def main():
    traning_data_file = f"data/processed/ms_marco_training_data_{max_lines}_lines_minfreq_{min_frequency}.json"
    
    with open(traning_data_file, "r") as f:
        training_data = json.load(f)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 32

    dFoo = dataset.MarcoTWOTOWERS(training_data)
    dataloader = torch.utils.data.DataLoader(dFoo, batch_size=batch_size, shuffle=True)
    
    docTower = DocTower(embedding_dim)
    queryTower = QueryTower(embedding_dim)

    docTower.to(dev)
    queryTower.to(dev)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(list(docTower.parameters()) + list(queryTower.parameters()), lr=learning_rate)

    epochs = 5

    # Initialize wandb
    wandb.init(
        project = "mlx7-week2-twotowers",
        name = f"{ts}",
        config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "embedding_dim": embedding_dim,
            "contrastive_loss_margin": contrastive_loss_margin,
        }
    )

    # make a folder in data/checkpoints with the current date and time

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%M_%S")
    os.makedirs(f"data/checkpoints/{dt_string}/two_towers", exist_ok=True)
    
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

            # Log batch loss to wandb
            wandb.log({"batch_loss": loss.item()})
    
        # Save model checkpoints
        doc_tower_path = f"data/checkpoints/{dt_string}/two_towers/doc_tower_epoch_{epoch+1}.pth"
        query_tower_path = f"data/checkpoints/{dt_string}/two_towers/query_tower_epoch_{epoch+1}.pth"
        
        torch.save(docTower.state_dict(), doc_tower_path)
        torch.save(queryTower.state_dict(), query_tower_path)

        # Log epoch loss to wandb
        epoch_loss = total_loss / len(dataloader)
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

        # Upload the files to Hugging Face
        hf_utils.save(
            [doc_tower_path, query_tower_path],
            repo_id,
            commit_message=f"Upload model checkpoints for epoch {epoch+1}"
        )

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()