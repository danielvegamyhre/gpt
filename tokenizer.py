import os
import string
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders

def clean_text(*files: str):
    output_files = []
    for file in files:
        # only clean the data if the clean file doesn't exist yet
        clean_file = file + '_clean' 
        if not os.path.exists(clean_file):
            # otherwise, perform the cleaning operation
            with open(file, 'r') as f:
                txt = f.read()
            txt = ''.join(c for c in txt if c not in string.punctuation or c == '.')
            txt = txt.encode("utf8").decode("ascii",'ignore')
            with open(clean_file, 'w') as f:
                f.write(txt)
        output_files.append(clean_file)
    
    print("Cleaned files: ", output_files)
    return output_files

def get_tokenizer(train: str, eval: str, tokenizer_file: str) -> Tokenizer:
    if os.path.exists(tokenizer_file):
        return Tokenizer.from_file(tokenizer_file)
    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    bert_tokenizer.pre_tokenizer = BertPreTokenizer()
    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    bert_tokenizer.train([train, eval], trainer)
    bert_tokenizer.decoder = decoders.WordPiece()
    bert_tokenizer.save(tokenizer_file)
    return bert_tokenizer
    
