import torch

class MarcoCBOW(torch.utils.data.Dataset):
  def __init__(self, corpus, vocab_to_int, int_to_vocab):
    self.corpus = corpus
    self.vocab_to_int = vocab_to_int
    self.int_to_vocab = int_to_vocab
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

# Test Suite
if __name__ == '__main__':
  ds = Marco()
  print(ds.tokens[:15])
  # print(ds[0])
  print(ds[5])
  dl = torch.utils.data.DataLoader(dataset=ds, batch_size=3)
  ex = next(iter(dl))
  print(ex)

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