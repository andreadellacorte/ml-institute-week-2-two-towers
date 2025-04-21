
import torch
from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokenised_dataset, get_corpus

dataset_id = "microsoft/ms_marco"  # Replace with your input file path
dataset_version = "v1.1"  # Replace with your desired dataset version
max_lines = 500
min_frequency = None  # Set the minimum frequency for tokenization

class Marco(torch.utils.data.Dataset):
  def __init__(self):
    dataset = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dataset)
    self.vocab_to_int, self.int_to_vocab = get_tokenised_dataset(clean_dataset, min_frequency)
    self.corpus = get_corpus(clean_dataset)
    self.tokens = [self.vocab_to_int[word] for word in self.corpus]

  def __len__(self):
    return len(self.tokens)

  def __getitem__(self, idx: int):
    ipt = self.tokens[idx]
    prv = self.tokens[idx-2:idx]
    nex = self.tokens[idx+1:idx+3]
    if len(prv) < 2: prv = [0] * (2 - len(prv)) + prv
    if len(nex) < 2: nex = nex + [0] * (2 - len(nex))
    return torch.tensor(prv + nex), torch.tensor([ipt])

if __name__ == '__main__':
  ds = Marco()
  print(ds.tokens[:15])
  # print(ds[0])
  print(ds[5])
  dl = torch.utils.data.DataLoader(dataset=ds, batch_size=3)
  ex = next(iter(dl))
  print(ex)