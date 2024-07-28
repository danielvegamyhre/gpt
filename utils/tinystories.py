from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

print(dataset)

train = dataset['train']
eval = dataset['validation']

with open('../data/tinystories/train.txt', 'w') as f:
    for i in range(len(train)//10):
        f.write(train[i]['text'])

with open('../data/tinystories/eval.txt', 'w') as f:
    for i in range(len(eval)//10):
        f.write(eval[i]['text'])