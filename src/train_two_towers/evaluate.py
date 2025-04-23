import torch
import json
from train_word2vec.model import CBOW
from doc_tower import DocTower
from query_tower import QueryTower
from sortedcontainers import SortedDict

from utils.loss_utils import contrastive_loss

clean_dataset_file = "data/processed/ms_marco_clean_500_lines.json"
word_to_ids_file = "data/processed/ms_marco_tkn_word_to_ids_500_lines_minfreq_0.json"
word_2vec_model_file = "data/checkpoints/cbow.500lines.256embeddings.0minfreq.5epochs.pth"
doc_tower_model_file = "data/checkpoints/2025_04_23__11_38_41/doc_tower/doc_tower_epoch_5.pth"
query_tower_model_file = "data/checkpoints/2025_04_23__11_38_41/query_tower/query_tower_epoch_5.pth"

embedding_dim = 256
contrastive_loss_margin = 0.2

def topk(mFoo, vocab_to_int, int_to_vocab):

  idx = vocab_to_int['computer']
  vec = mFoo.emb.weight[idx].detach()
  with torch.no_grad():

    vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1)
    emb = torch.nn.functional.normalize(mFoo.emb.weight.detach(), p=2, dim=1)
    sim = torch.matmul(emb, vec.squeeze())
    top_val, top_idx = torch.topk(sim, 6)
    print('\nTop 5 words similar to "computer":')
    count = 0
    for i, idx in enumerate(top_idx):
      word = int_to_vocab[idx.item()]
      sim = top_val[i].item()
      print(f'  {word}: {sim:.4f}')
      count += 1
      if count == 5: break

def feed(query, doc, queryModel, docModel):
  # feed all services to the model and return the output
  query = queryModel(query)
  doc = docModel(doc)

  return contrastive_loss(query, doc, doc, contrastive_loss_margin)

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
  
def main():
  
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
    for i, row in enumerate(clean_dataset):
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

  example_query = "pork products"

  query_embedding = sentence_to_mean_embedding(example_query, cbow, vocab_to_int, dev)

  query_tower_output = query_tower(query_embedding)
  
  sorted_dictionary = SortedDict()

  passage_embeddings = torch.cat([passage['passage_embedding'] for passage in passages], dim=0)
  doc_tower_outputs = doc_tower(passage_embeddings)
  # calculate the cosine similarity between the query and all passages
  similarity = torch.nn.functional.cosine_similarity(query_tower_output, doc_tower_outputs, dim=1)
  # add the passages to the sorted dictionary
  for i, contrastive_loss_value in enumerate(similarity):
    sorted_dictionary[contrastive_loss_value.item()] = passages[i]

  
    
  # print the top 5 passages
  print("Top 5 passages for query:", example_query)
  for i, (key, value) in enumerate(reversed(sorted_dictionary.items())):
    if i == 5:
      break
    print(f"Passage {i+1}: {value['passage']}")
    print(f"Cosine similarity: {key}")

if __name__ == "__main__":
  main()