#!/usr/bin/python3
"""
Preprocessing script for c4 dataset to convert from JSONL to raw text.
"""
import os
import json
import argparse

def main(input_file: str, train_file: str, eval_file: str, train_split: float) -> None:
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"input file '{input_file}' does not exist")

    print(f"processing input file: {input_file}")
    with open(input_file, 'r') as json_file:
        json_list: list[str] = list(json_file)

        # create train/eval split
        total_samples: int = len(json_list)
        train_samples: int = int(total_samples * train_split)
        
        print(f"writing to training file: {train_file}")
        with open(train_file, 'a') as f:
            for i in range(train_samples):
                json_str: str  = json_list[i]
                sample: dict = json.loads(json_str)
                text: str = sample['text']
                # append newline to sample if not present to keep samples separated.
                if not text.endswith('\n'):
                    text += '\n'
                f.write(text)

        print(f"writing to eval file: {eval_file}")
        with open(eval_file, 'a') as f:
            for i in range(train_samples, total_samples):
                json_str: str  = json_list[i]
                sample: dict = json.loads(json_str)
                text: str = sample['text']
                # append newline to sample if not present to keep samples separated.
                if not text.endswith('\n'):
                    text += '\n'
                f.write(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parser to convert c4 dataset from .jsonl format to raw text") 
    parser.add_argument("--input-file", type=str, help="input file of c4 dataset", required=True)
    parser.add_argument("--train-file", type=str, help="output file for training data", required=True)
    parser.add_argument("--eval-file", type=str, help="output file for eval data", required=True)
    parser.add_argument("--train-split", type=float, default=0.8, help="fraction of dataset to parse into the training dataset (e.g. 0.8). The rest will be parsed into the eval dataset.")
    args = parser.parse_args()
    main(args.input_file, args.train_file, args.eval_file, args.train_split)