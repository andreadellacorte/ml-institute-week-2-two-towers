import torch
import random

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

class MarcoTWOTOWERS(torch.utils.data.Dataset):
    def __init__(self, data, vocab_to_int, embs):
        self.data = data
        self.embs = embs
        self.vocab_to_int = vocab_to_int

    def preprocess(self, text):

      text = text.lower()

      # text = text.replace('.',  ' <PERIOD> ')
      # text = text.replace(',',  ' <COMMA> ')
      # text = text.replace('"',  ' <QUOTATION_MARK> ')
      # text = text.replace('“',  ' <QUOTATION_MARK> ')
      # text = text.replace('”',  ' <QUOTATION_MARK> ')
      # text = text.replace(';',  ' <SEMICOLON> ')
      # text = text.replace('!',  ' <EXCLAMATION_MARK> ')
      # text = text.replace('?',  ' <QUESTION_MARK> ')
      # text = text.replace('(',  ' <LEFT_PAREN> ')
      # text = text.replace(')',  ' <RIGHT_PAREN> ')
      # text = text.replace('--', ' <HYPHENS> ')
      # text = text.replace('?',  ' <QUESTION_MARK> ')
      # text = text.replace(':',  ' <COLON> ')
      # text = text.replace("'",  ' <APOSTROPHE> ')
      # text = text.replace("’",  ' <APOSTROPHE> ')

      words = text.split()

      for i, word in enumerate(words):
          # Remove punctuation
          if word in self.vocab_to_int:
              words[i] = word
          else:
              words[i] = '<UNK>'

      return words

    def to_emb(self, text):
      text = self.preprocess(text)
      tkns = [self.vocab_to_int[t] for t in text if t in self.vocab_to_int]
      if len(tkns) == 0: return
      tkns = torch.tensor(tkns).to('cuda:0')
      embs = self.embs(tkns)
      return embs.mean(dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = []

        row = self.data[idx]

        # Ensure there are relevant passages
        if not any(row["is_selected"]):
            raise ValueError(f"No relevant passages found for index {idx}")

        for i, is_selected in enumerate(row["is_selected"]):
            if not is_selected:
                continue

            relevant_passage = row["passages"][i]

            for _ in range(10):
                while True:
                    irrelevant_row = random.choice(self.data)
                    if irrelevant_row != row:
                        break

                irrelevant_passage = random.choice(irrelevant_row["passages"])

                query_emb = self.to_emb(row["query"])
                relevant_emb = self.to_emb(relevant_passage)
                irrelevant_emb = self.to_emb(irrelevant_passage)

                # Ensure embeddings are not None
                if query_emb is None or relevant_emb is None or irrelevant_emb is None:
                    continue

                batch.append((query_emb, relevant_emb, irrelevant_emb))

            break

        if not batch:
            raise ValueError(f"No valid data found for index {idx}")

        return batch
  
# Test Suite
if __name__ == '__main__':
  ds = MarcoCBOW()
  print(ds.tokens[:15])
  # print(ds[0])
  print(ds[5])
  dl = torch.utils.data.DataLoader(dataset=ds, batch_size=3)
  ex = next(iter(dl))
  print(ex)