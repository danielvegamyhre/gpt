#!/usr/bin/python3
import os
import torch
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt

from gpt import GPT, SEQ_LEN
from tokenizer import get_tokenizer, clean_text, Tokenizer
import tiktoken

# hyperparameters
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
EVAL_INTERVAL = 5
EVAL_ITERS = 1
TRAIN = 'train'
EVAL = 'eval'
DATA = {TRAIN: '', EVAL: ''}


def plot_losses(train_losses: list[float], eval_losses: list[float]):
    """
    Plots the training loss and validation loss.

    Parameters:
    train_losses (list of float): List of training loss values.
    eval_losses (list of float): List of evaluation loss values.
    """
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(train_losses, label='Training Loss', color='blue')
    
    # Plot validation loss
    plt.plot(eval_losses, label='Validation Loss', color='orange')
    
    # Add title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.show()

def get_batch(split: str, batch_size: int = 1):
    data = DATA[split]
    ix = torch.randint(len(data) - SEQ_LEN, (batch_size,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([data[i+1:i+1+SEQ_LEN] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size: int = 1):
    out = {}
    model.eval()
    for split in [TRAIN, EVAL]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split, batch_size)
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

def load_checkpoint(path, model, optim):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'loaded checkpoint from {path} at epoch {epoch}')
    return epoch, loss

def preprocess_data(train: str, eval: str, tokenizer):
    train_out = f'{train}_tokenized' 
    eval_out = f'{eval}_tokenized' 

    # load preprocessed train data if we have it, otherwise preprocess the raw dataset
    if os.path.exists(train_out):
        DATA[TRAIN] = torch.load(train_out)
    else:
        print(f'tokenizing training data: {train}')
        with open(train) as f:
            text = f.read()
         
        ids = tokenizer.encode(text)
        DATA[TRAIN] = torch.tensor(ids, dtype=torch.long, device=DEVICE)

        # store pre-processed training data 
        torch.save(DATA[TRAIN], train_out)

    # load preprocessed eval data if we have it, otherwise preprocess the raw dataset
    if os.path.exists(eval_out):
        DATA[EVAL] = torch.load(eval_out)
    else:
        print(f'tokenizing eval data: {eval}')
        with open(eval) as f:
            text = f.read()
            
        # tokenize training data
        ids = tokenizer.encode(text)
        DATA[EVAL] = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        
        # store pre-processed training data 
        torch.save(DATA[EVAL], eval_out)

def main(args: argparse.Namespace):
    print(f'device: {DEVICE}')

    # clean text
    print('cleaning data')
    cleaned_train, cleaned_eval = clean_text(args.train, args.eval)

    # train tokenizer
    print('training tokenizer')
    tokenizer = tiktoken.get_encoding('o200k_base') # get_tokenizer(cleaned_train, cleaned_eval, args.tokenizer)

    print('tokenizing training and eval data')
    preprocess_data(cleaned_train, cleaned_eval, tokenizer)
    print('finished loading preprocessed data')

    # create model
    vocab_size = tokenizer.n_vocab
    model = GPT(vocab_size)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print('Vocab size: ', vocab_size)
    model = model.to(DEVICE)
    
    # create optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    epoch, loss = 0, float('inf')

    # load checkpoint if specified
    if args.load_checkpoint:
        if not os.path.isfile(args.load_checkpoint):
            raise FileNotFoundError(f"checkpoint file does not exist: {args.load_checkpoint}")
        epoch, loss = load_checkpoint(args.load_checkpoint, model, optim)
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'loaded checkpoint from {args.load_checkpoint} at epoch {epoch}')

    # if we are just generating output and not training
    if args.generate:
        print(f'generating output of length {args.generate}')
        context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
        generate = lambda n: tokenizer.decode(model.generate(idx=context, max_new_tokens=n)[0].tolist())
        print(generate(args.generate))
        return 
    
    # training loop
    print('starting training')
    train_losses, eval_losses = [], []
    for i in tqdm(range(epoch, args.epochs)):
        # don't estimate on first epoch
        if i % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, args.batch_size)
            print(f"step {i} train loss {losses[TRAIN]} eval loss {losses[EVAL]}")
            train_losses.append(losses[TRAIN])
            eval_losses.append(losses[EVAL])

        xb, yb = get_batch(TRAIN, args.batch_size)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        # don't checkpoint on first epoch even if it is divisible by checkpoint interval,
        # and always checkpoint after the last epoch.
        if args.save_checkpoint != "" and (i != epoch and i % args.checkpoint_interval == 0):
            save_checkpoint(args.save_checkpoint, i, loss, model, optim)


    # always save checkpoint before exiting
    save_checkpoint(args.save_checkpoint, i, loss, model, optim) 

    plot_losses(train_losses, eval_losses)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str, required=True, help='training data file')
    argparser.add_argument('--eval', type=str, required=True, help='eval data file')
    argparser.add_argument('--generate', type=int, help='generate output of length N')
    argparser.add_argument('--tokenizer', type=str, help="tokenizer JSON file", default="tokenizer/default.json")
    argparser.add_argument('--save-checkpoint', type=str, help="file to checkpoint model parameters to", default="checkpoints/default-checkpoint.pt")
    argparser.add_argument('--load-checkpoint', type=str, help="file to load model parameters from")
    argparser.add_argument('--checkpoint-interval', type=int, help="number of epochs between checkpoints", default=1000)
    argparser.add_argument('--epochs', type=int, help='number of training epochs', default=0)
    argparser.add_argument('--lr', type=float, help="learning rate", default=1e-3)
    argparser.add_argument('--batch-size', type=int, help="batch size", default=1)
    args = argparser.parse_args()
    main(args)
