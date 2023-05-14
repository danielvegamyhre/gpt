#!/usr/bin/python3
import os
import torch
import argparse
from models.gpt import GPT, BATCH_SIZE, SEQ_LEN
from tqdm import tqdm

# hyperparameters
MAX_ITERS = 5000
LEARNING_RATE = 1e-3
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
EVAL_INTERVAL = 1000
EVAL_ITERS = 100
TRAIN = 'train'
EVAL = 'eval'
DATA = {TRAIN: '', EVAL: ''}
CHECKPOINT_PATH = 'model.pt'
CHECKPOINT_INTERVAL = 1000

def get_batch(split):
    data = DATA[split]
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([data[i+1:i+1+SEQ_LEN] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in [TRAIN, EVAL]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

def save_checkpoint(epoch, loss, model, optim):
    print(f'checkpointing at epoch {epoch}')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            }, CHECKPOINT_PATH)

def load_checkpoint(model, optim):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'loaded checkpoint from {CHECKPOINT_PATH} at epoch {epoch}')
    return epoch, loss


def main(args):
    # obtain character vocab from data
    with open(args.data, 'r') as fp:
        text = fp.read()
    chars = sorted(list(set(text)))

    # creating numerical encoding for chars
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda n: ''.join(itos[i] for i in n)

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    DATA[TRAIN] = data[:n]
    DATA[EVAL] = data[n:]

    # create model
    model = GPT(len(chars))
    m = model.to(DEVICE)
    
    # create optimizer
    optim = torch.optim.AdamW(m.parameters(), lr=1e-3)
    epoch, loss = 0, float('inf')

    # load from checkpoint if it exists
    if os.path.isfile(CHECKPOINT_PATH):
        epoch, loss =load_checkpoint(model, optim)

    # if we are just generating output and not training
    if args.generate:
        print(f'generating output of length {args.generate}')
        context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
        generate = lambda n: decode(m.generate(idx=context, max_new_tokens=n)[0].tolist())
        print(generate(300))
        return

    # training loop
    for i in tqdm(range(epoch, MAX_ITERS)):
        if i % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {i} train loss {losses['train']} eval loss {losses['eval']}")

        xb, yb = get_batch(TRAIN)
        _, loss = m(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if i != epoch and i % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(i, loss, m, optim)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, help='train <file>: train GPT on dataset', required=False)
    argparser.add_argument('--generate', type=int, help='generate <N>: generate output of length N')
    args = argparser.parse_args()
    main(args)
