from pathlib import Path
from argparse import ArgumentParser
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_folder', '-df', required=True, help='Path to the folder containing all txt files for the corpus.')
    parser.add_argument('--out_folder', '-of', required=True, help='Path to the folder where to save the tokenizer.')
    parser.add_argument('--vocab_size', '-vs', default=52_000, help='Number of unique tokens in the vocabulary.')
    parser.add_argument('--min_freq', '-mf', default=2, help='Minimum frequency to consider a token as valid.')
    args = parser.parse_args()

    folder_path = args.data_folder
    vocab_size = args.vocab_size
    min_freq = args.min_freq
    out_folder = args.out_folder
    paths = [str(x) for x in Path(folder_path).glob("**/*.txt")]

    # Initialize a tokenizer    
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer.save_model(out_folder)
