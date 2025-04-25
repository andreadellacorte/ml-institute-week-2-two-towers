from config import *

import torch
import json
from tqdm import tqdm  # Import tqdm for the progress bar
from model import CBOW
from model import DocTower
from model import QueryTower
from sortedcontainers import SortedDict

from utils.loss_utils import contrastive_loss

clean_dataset_file = f"data/processed/ms_marco_clean_{max_lines}_lines.json"
word_to_ids_file = f"data/processed/ms_marco_tkn_word_to_ids_{max_lines}_lines_minfreq_{min_frequency}.json"
word_2vec_model_file = f"data/checkpoints/cbow.{max_lines}lines.256embeddings.{min_frequency}minfreq.5epochs.pth"
doc_tower_model_file = "data/checkpoints/2025_04_23__11_38_41/doc_tower/doc_tower_epoch_5.pth"
query_tower_model_file = "data/checkpoints/2025_04_23__11_38_41/query_tower/query_tower_epoch_5.pth"

def sentence_to_mean_embedding(sentence, word2vec, vocab_to_int, device):
  tokenised_sentence = [vocab_to_int[word] for word in sentence.split()]

  for i in range(len(tokenised_sentence)):
    tokenised_sentence[i] = word2vec.emb.weight[tokenised_sentence[i]].detach()
    # average the embeddings
  
  tokenised_sentence = torch.stack(tokenised_sentence)
  tokenised_sentence = torch.mean(tokenised_sentence, dim=0)
  tokenised_sentence = tokenised_sentence.unsqueeze(0)
  tokenised_sentence = tokenised_sentence.to(device)

  return tokenised_sentence

def evaluate_example_queries(example_queries, passages, cbow, vocab_to_int, dev, query_tower, doc_tower, k=5):
    batch_size = 32  # Customize the batch size here

    for example_query in example_queries:
      query_embedding = sentence_to_mean_embedding(example_query, cbow, vocab_to_int, dev)
      query_tower_output = query_tower(query_embedding)
      sorted_dictionary = SortedDict()

      for i in range(0, len(passages), batch_size):
        batch_passages = passages[i:i+batch_size]
        passage_embeddings = torch.cat([passage['passage_embedding'] for passage in batch_passages], dim=0)
        doc_tower_outputs = doc_tower(passage_embeddings)
        # calculate the cosine similarity between the query and all passages in the batch
        similarity = torch.nn.functional.cosine_similarity(query_tower_output, doc_tower_outputs, dim=1)
        # add the passages to the sorted dictionary
        for j, sim in enumerate(similarity):
          sorted_dictionary[sim.item()] = batch_passages[j]

      # print the top 5 passages
      print("Top 5 passages for query:", example_query)
      for i, (key, value) in enumerate(reversed(sorted_dictionary.items())):
        if i == 5:
          break
        print(f"Passage {i+1}: {value['passage']}")
        print(f"Cosine similarity: {key}")

def calculate_metrics(queries, passages, query_tower, doc_tower, k=10):
    total_reciprocal_rank = 0
    total_average_precision = 0
    total_average_recall = 0
    total_queries = len(queries)

    print(f"Calculating Mean Reciprocal Rank (MRR), Average Recall (MAR) and Average Precision (MAP) at k = {k} for all queries:")

    for query_embedding, relevant_passages in tqdm(queries.items(), desc="Evaluating queries"):
        query_tower_output = query_tower(query_embedding)
        sorted_dictionary = SortedDict()

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i+batch_size]
            passage_embeddings = torch.cat([passage['passage_embedding'] for passage in batch_passages], dim=0)
            doc_tower_outputs = doc_tower(passage_embeddings)
            similarity = torch.nn.functional.cosine_similarity(query_tower_output, doc_tower_outputs, dim=1)

            for j, sim in enumerate(similarity):
                sorted_dictionary[sim.item()] = batch_passages[j]

        # Calculate reciprocal rank for the current query
        for rank, (key, value) in enumerate(reversed(sorted_dictionary.items()), start=1):
            if value['relevant_query'] == query_embedding:
                total_reciprocal_rank += 1 / rank
                break

        # Calculate precision and recall at k
        relevant_count = 0
        precision_at_k = 0
        recall_at_k = 0

        for rank, (key, value) in enumerate(reversed(sorted_dictionary.items()), start=1):
            if rank > k:
                break
            if value['relevant_query'] == query_embedding:
                relevant_count += 1
                precision_at_k += relevant_count / rank

        total_relevant = len(relevant_passages)
        recall_at_k = relevant_count / total_relevant if total_relevant > 0 else 0
        total_average_precision += precision_at_k / k
        total_average_recall += recall_at_k

    mean_reciprocal_rank = total_reciprocal_rank / total_queries
    mean_average_precision = total_average_precision / total_queries
    mean_average_recall = total_average_recall / total_queries

    print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank}")
    print(f"Mean Average Precision (MAP): {mean_average_precision}")
    print(f"Mean Average Recall (MAR): {mean_average_recall}")

def evaluate_two_towers():
  
  with open(word_to_ids_file, "r") as f:
    vocab_to_int = json.load(f)

  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load the word2vec model
  cbow = CBOW(len(vocab_to_int), embedding_dim)
  cbow.load_state_dict(torch.load(word_2vec_model_file))
  cbow.eval()
  cbow.to(dev)

  # Load the doc tower model
  doc_tower = DocTower(embedding_dim)
  doc_tower.load_state_dict(torch.load(doc_tower_model_file))
  doc_tower.eval()
  doc_tower.to(dev)

  # Load the query tower model
  query_tower = QueryTower(embedding_dim)
  query_tower.load_state_dict(torch.load(query_tower_model_file))
  query_tower.eval()
  query_tower.to(dev)

  with open(clean_dataset_file, "r") as f:
    clean_dataset = json.load(f)

  queries = {}
  passages = []

  with torch.no_grad():
    for i, row in enumerate(tqdm(clean_dataset, desc="Processing dataset")):
      query_embedding = sentence_to_mean_embedding(row["query"], cbow, vocab_to_int, dev)
      queries[query_embedding] = []
      for passage in row["passages"]:
        passage_embedding = sentence_to_mean_embedding(passage, cbow, vocab_to_int, dev)
        passages.append({
          'relevant_query': row["query"],
          'passage': passage,
          'relevant_query_embedding': query_embedding,
          'passage_embedding': passage_embedding
        })
        queries[query_embedding].append(passage_embedding)

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
  evaluate_example_queries(example_queries, passages, cbow, vocab_to_int, dev, query_tower, doc_tower, k=5)

  calculate_metrics(queries, passages, query_tower, doc_tower, k=10)

if __name__ == "__main__":
  evaluate_two_towers()