#
#
#
import torch
from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokenised_dataset


dataset_id = "microsoft/ms_marco"  # Replace with your input file path
dataset_version = "v1.1"  # Replace with your desired dataset version
max_lines = 500
min_frequency = None  # Set the minimum frequency for tokenization

#
#
dataset = get_raw_dataset(dataset_id, dataset_version, max_lines)
clean_dataset = get_clean_dataset(dataset)
vocab_to_int, int_to_vocab = get_tokenised_dataset(clean_dataset, min_frequency)

#
#
#
def topk(mFoo):

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