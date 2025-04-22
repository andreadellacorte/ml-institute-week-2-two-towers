import tqdm
import wandb
import torch
import dataset
import evaluate
import datetime
import sys
import os
import model as model

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokenised_dataset, get_corpus

dataset_id = "microsoft/ms_marco"
dataset_version = "v1.1"
max_lines = 500
min_frequency = 0
embedding_dim = 256
batch_size = 4096
learning_rate = 0.003

torch.manual_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

dFoo = get_raw_dataset(dataset_id, dataset_version, max_lines)
clean_dataset = get_clean_dataset(dFoo)
vocab_to_int, int_to_vocab = get_tokenised_dataset(clean_dataset, min_frequency)
corpus = get_corpus(clean_dataset, vocab_to_int, min_frequency)

ds = dataset.MarcoCBOW(corpus, vocab_to_int, int_to_vocab)
dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)

mFoo = model.CBOW(len(vocab_to_int), embedding_dim)
print('mFoo:params', sum(p.numel() for p in mFoo.parameters()))
opFoo = torch.optim.Adam(mFoo.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()


#
#
#
wandb.init(project='mlx7-week2-cbow', name=f'{ts}')

wandb.config.update({
    'dataset_id': dataset_id,
    'dataset_version': dataset_version,
    'max_lines': max_lines,
    'min_frequency': min_frequency,
    'embedding_dim': embedding_dim,
    'batch_size': batch_size,
    'learning_rate': learning_rate
})

mFoo.to(dev)

#
#
#
for epoch in range(5):
  prgs = tqdm.tqdm(dl, desc=f'Epoch {epoch+1}', leave=False)
  for i, (ipt, trg) in enumerate(prgs):
    ipt, trg = ipt.to(dev), trg.to(dev)
    opFoo.zero_grad()
    out = mFoo(ipt)
    loss = criterion(out, trg.squeeze())
    loss.backward()
    opFoo.step()
    wandb.log({'loss': loss.item()})
    if i % 100 == 0:
       prgs.set_postfix({'loss': loss.item()})  # Add loss to the loading bar
    if i % 10_000 == 0:
        evaluate.topk(mFoo, vocab_to_int, int_to_vocab)

  # checkpoint
  checkpoint_name = f'{ts}.cbow.{len(dFoo)}lines.{embedding_dim}embeddings.{min_frequency}minfreq.{epoch + 1}epochs.pth'
  torch.save(mFoo.state_dict(), f'./data/checkpoints/{checkpoint_name}')
  artifact = wandb.Artifact('model-weights', type='model')
  artifact.add_file(f'./data/checkpoints/{checkpoint_name}')
  wandb.log_artifact(artifact)


#
#
#
wandb.finish()