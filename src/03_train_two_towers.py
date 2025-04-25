from config import *

import os
from datetime import datetime

import torch
import tqdm

from utils import file_utils, hf_utils, data_utils

from dataset import MarcoTWOTOWERS
from model import CBOW
from model import DocTower
from model import QueryTower

import evaluate

from utils.loss_utils import contrastive_loss
import wandb  # Add wandb import

def main():
    ts = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16

    model_path = f"data/models/cbow.{max_lines}lines.{embedding_dim}embeddings.{min_frequency}minfreq.5epochs.pth"
    vocab_to_int = file_utils.load_json(f'data/processed/ms_marco_tkn_word_to_ids_{max_lines}_lines_minfreq_{min_frequency}.json')

    cbow = CBOW(len(vocab_to_int), embedding_dim)
    cbow.load_state_dict(torch.load(model_path))
    cbow.eval()
    
    cbow.to(dev)

    marcoData = data_utils.get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = data_utils.get_clean_dataset(marcoData)

    # remove from clean_dataset the queries not any(row["is_selected"]):
    clean_dataset = [row for row in clean_dataset if any(row["is_selected"])]

    dFoo = MarcoTWOTOWERS(clean_dataset, vocab_to_int, cbow.emb)

    def collate_fn(batch):
        queries, relevant_docs, irrelevant_docs = zip(*[item for sublist in batch for item in sublist])
        return torch.stack(queries), torch.stack(relevant_docs), torch.stack(irrelevant_docs)

    dataloader = torch.utils.data.DataLoader(dFoo, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    docTower = DocTower(embedding_dim)
    queryTower = QueryTower(embedding_dim)

    docTower.to(dev)
    queryTower.to(dev)

    learning_rate = 0.02
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

    evaluate_queries, evaluate_passages = evaluate.embed_queries_passages(cbow, vocab_to_int, dev, clean_dataset[:1000])
    
    for epoch in range(epochs):
        queryTower.train()
        docTower.train()
        prgs = tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
        total_loss = 0.0

        for i, (query, relevant_doc, irrelevant_doc) in enumerate(prgs):

            query = query.to(dev)
            relevant_doc = relevant_doc.to(dev)
            irrelevant_doc = irrelevant_doc.to(dev)

            query = torch.nn.functional.normalize(queryTower(query))
            pos = torch.nn.functional.normalize(docTower(relevant_doc))
            neg = torch.nn.functional.normalize(docTower(irrelevant_doc))
            
            loss = contrastive_loss(query, pos, neg, contrastive_loss_margin)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prgs.set_postfix(loss=total_loss / (prgs.n + 1))

            # Log additional metrics to wandb
            wandb.log({
                "batch_loss": loss.item(),
                "gradient_norm": sum(p.grad.norm().item() for p in docTower.parameters() if p.grad is not None),
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

            if i % 100 == 0:
                example_queries = [
                    "pork products",
                    "was ronald reagan a good president",
                    "who is the ronald reagan",
                    "who is the president of the united states",
                    "poverty",
                    "united states",
                    "united kingdom",
                ]

                # Call the method in the main function
                evaluate.evaluate_example_queries(example_queries, evaluate_passages, cbow, vocab_to_int, dev, queryTower, docTower, k=5)

                mean_reciprocal_rank, mean_average_precision, mean_average_recall = evaluate.calculate_metrics(evaluate_queries, evaluate_passages, queryTower, docTower, k=10)

                wandb.log({
                    "mean_reciprocal_rank": mean_reciprocal_rank,
                    "mean_average_precision": mean_average_precision,
                    "mean_average_recall": mean_average_recall,
                })

                print(f"MRR: {mean_reciprocal_rank}, MAP: {mean_average_precision}, MAR: {mean_average_recall}")

        # Save model checkpoints
        doc_tower_path = f"data/checkpoints/{dt_string}/two_towers/doc_tower_epoch_{epoch+1}.pth"
        query_tower_path = f"data/checkpoints/{dt_string}/two_towers/query_tower_epoch_{epoch+1}.pth"
        
        torch.save(docTower.state_dict(), doc_tower_path)
        torch.save(queryTower.state_dict(), query_tower_path)

        # Log embedding distributions
        wandb.log({
            "query_embedding_distribution": wandb.Histogram(query.cpu().detach().numpy()),
            "relevant_doc_embedding_distribution": wandb.Histogram(pos.cpu().detach().numpy()),
            "irrelevant_doc_embedding_distribution": wandb.Histogram(neg.cpu().detach().numpy()),
        })

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