#!/usr/bin/python3
import os
import torch
import argparse
from models.gpt import GPT, BATCH_SIZE, SEQ_LEN
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


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

def train_tokenizer():
    output_file ="data/tokenizer-wiki.json"
    if os.path.exists(output_file):
        print(f'loading tokenizer from {output_file}')
        tokenizer = Tokenizer.from_file(output_file)
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = Whitespace()
        files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
        tokenizer.train(files, trainer)
        print(f'saving tokenizer to {output_file}')
        tokenizer.save(output_file)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    return tokenizer


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
    print(f'DEVICE: {DEVICE}')
    with open(args.data) as f:
        text = f.read()
    tokenizer = train_tokenizer()

    # creating numerical encoding for chars
    encode = tokenizer.encode
    decode = tokenizer.decode

    data = torch.tensor(encode(text).ids, dtype=torch.long, device=DEVICE)

    n = int(0.9 * len(data))
    DATA[TRAIN] = data[:n]
    DATA[EVAL] = data[n:]

    # create model
    model = GPT(tokenizer.get_vocab_size())
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
    print('starting training')
    for i in tqdm(range(epoch, MAX_ITERS)):
        if i != epoch and i % EVAL_INTERVAL == 0:
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
