import copy
import json
import tqdm
from datasets import load_dataset, concatenate_datasets
import os
import argparse
import re
from data import DATA_DIR


def prepare_dataset():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset_name', default='danish_legal_pile')
    parser.add_argument('--max_lengths', default=[128, 512, 2048])
    config = parser.parse_args()

    CUSTOM_DATA_FOLDER = os.path.join(DATA_DIR, config.dataset_name)

    if not os.path.exists(CUSTOM_DATA_FOLDER):
        os.mkdir(CUSTOM_DATA_FOLDER)

    # MERGE DANISH LEGAL AND EU CORPORA
    danish_dataset = load_dataset('DDSC/partial-danish-gigaword-no-twitter', split='train')
    danish_eurlex_dataset = load_dataset('multi_eurlex', 'da', split='train')

    if config.dataset_name == 'danish_legal_pile':
        danish_law_dataset = danish_dataset.filter(
            lambda example: example['source'] in ['retsinformationdk', 'retspraksis'])
        dataset = concatenate_datasets([danish_law_dataset, danish_eurlex_dataset])
        dataset = dataset.shuffle(seed=42)
    else:
        dataset = concatenate_datasets([danish_dataset, danish_eurlex_dataset])
        dataset = dataset.shuffle(seed=42)

    # SPLIT INTO TRAIN/TEST
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # CREATE DERIVATIVES WITH SHORTER LENGTH
    for size in config.max_lengths:
        max_sw_seq_length = int(size * 0.9)
        for split in ['train', 'test']:
            with open(os.path.join(CUSTOM_DATA_FOLDER, f'{split}_{size}.jsonl'), mode='w', encoding='utf8') as file:
                for line in tqdm.tqdm(dataset[split]['text']):
                    line = re.sub(r'(\s)+', r'\1', line)
                    if len(line.split(' ')) > int(size/4) and not line.isspace():
                        ws_tokens = line.split(' ')
                        prev_idx = 0
                        for idx in range(max_sw_seq_length, len(ws_tokens) + max_sw_seq_length, max_sw_seq_length):
                            chunk = copy.deepcopy(' '.join(ws_tokens[prev_idx:idx]))
                            if len(chunk.split(' ')) > int(size/4) and not chunk.isspace():
                                file.write(json.dumps({'text': chunk}) + '\n')
                                prev_idx = idx


if __name__ == '__main__':
    prepare_dataset()
