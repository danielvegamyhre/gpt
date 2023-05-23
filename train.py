#!/usr/bin/python3
import os
import torch
import argparse
from gpt import GPT, BATCH_SIZE, SEQ_LEN
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


# hyperparameters
LEARNING_RATE = 1e-3
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
EVAL_INTERVAL = 1000
EVAL_ITERS = 100
TRAIN = 'train'
EVAL = 'eval'
DATA = {TRAIN: '', EVAL: ''}
CHECKPOINT_INTERVAL = 1000

def get_tokenizer(train: str, eval: str, tokenizer_file: str) -> Tokenizer:
    if os.path.exists(tokenizer_file):
        print(f'loading tokenizer from {tokenizer_file}')
        tokenizer = Tokenizer.from_file(tokenizer_file)
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = Whitespace()
        files = [train, eval]
        tokenizer.train(files, trainer)
        print(f'saving tokenizer to {tokenizer_file}')
        tokenizer.save(tokenizer_file)
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

def save_checkpoint(path, epoch, loss, model, optim):
    print(f'checkpointing at epoch {epoch}')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            }, path)

def load_checkpoint(model, optim):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'loaded checkpoint from {CHECKPOINT_PATH} at epoch {epoch}')
    return epoch, loss

def preprocess_data(train: str, eval: str, tokenizer: Tokenizer):
    train_out = f'{train}_tokenized' 
    eval_out = f'{eval}_tokenized' 

    # load preprocessed data if we have it, otherwise preprocess the raw dataset
    if os.path.exists(train_out):
        DATA[TRAIN] = torch.load(train_out)
    else:
        print(f'tokenizing traing data: {train}')
        with open(args.train) as f:
            text = f.read()
            
        # tokenize training data
        DATA[TRAIN] = tokenizer.encode(text).ids

        # store pre-processed training data 
        torch.save(torch.tensor(DATA[TRAIN], dtype=torch.long, device=DEVICE), train_out)

    # load preprocessed data if we have it, otherwise preprocess the raw dataset
    if os.path.exists(eval_out):
        DATA[EVAL] = torch.load(eval_out)
    else:
        print(f'tokenzing eval data: {eval}')
        with open(args.eval) as f:
            text = f.read()
            
        # tokenize training data
        DATA[EVAL] = tokenizer.encode(text).ids
        
        # store pre-processed training data 
        torch.save(torch.tensor(DATA[EVAL], dtype=torch.long, device=DEVICE), eval_out)

def main(args: argparse.Namespace):
    print(f'device: {DEVICE}')

    # train tokenizer
    tokenizer = get_tokenizer(args.train, args.eval, args.tokenizer)

    preprocess_data(args.train, args.eval, tokenizer)
    print('finished loading preprocessed data')

    # create model
    model = GPT(tokenizer.get_vocab_size())
    m = model.to(DEVICE)
    
    # create optimizer
    optim = torch.optim.AdamW(m.parameters(), lr=1e-3)
    epoch, loss = 0, float('inf')

    # load from checkpoint if it exists
    if os.path.isfile(args.checkpoint):
        epoch, loss = load_checkpoint(args.checkpoint, model, optim)

    # if we are just generating output and not training
    if args.generate:
        print(f'generating output of length {args.generate}')
        context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
        generate = lambda n: tokenizer.decode(m.generate(idx=context, max_new_tokens=n)[0].tolist())
        print(generate(300))
        return

    # training loop
    print('starting training')
    for i in tqdm(range(epoch, args.epochs)):
        if i != epoch and i % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {i} train loss {losses['train']} eval loss {losses['eval']}")

        xb, yb = get_batch(TRAIN)
        _, loss = m(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if i != epoch and i % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(args.checkpoint, i, loss, m, optim)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str, help='training data file')
    argparser.add_argument('--eval', type=str, help='eval data file')
    argparser.add_argument('--generate', type=int, help='generate output of length N')
    argparser.add_argument('--tokenizer', type=str, help="tokenizer JSON file", default="tokenizer/default.json")
    argparser.add_argument('--checkpoint', type=str, help="model checkpoint file")
    argparser.add_argument('--epochs', type=int, help='number of training epochs')
    args = argparser.parse_args()
    main(args)
