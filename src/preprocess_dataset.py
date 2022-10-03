import json

import tqdm
from datasets import load_dataset, concatenate_datasets
import os
from data import DATA_DIR


CUSTOM_DATA_FOLDER = os.path.join(DATA_DIR, 'danish_legal_pile')

if not os.path.exists(CUSTOM_DATA_FOLDER):
    os.mkdir(CUSTOM_DATA_FOLDER)

# MERGE DANISH LEGAL AND EU CORPORA
danish_dataset = load_dataset('DDSC/partial-danish-gigaword-no-twitter', split='train')
danish_eurlex_dataset = load_dataset('multi_eurlex', 'da', split='train')
danish_law_dataset = danish_dataset.filter(
    lambda example: example['source'] in ['retsinformationdk', 'retspraksis'])
danish_legal_dataset = concatenate_datasets([danish_law_dataset, danish_eurlex_dataset])
danish_legal_dataset = danish_legal_dataset.shuffle(seed=42)

# SPLIT INTO TRAIN/TEST
danish_legal_dataset = danish_legal_dataset.train_test_split(test_size=0.1, seed=42)

# CREATE DERIVATIVES WITH SHORTER LENGTH
for size in [128, 512]:
    max_sw_seq_length = int(size * 0.9)
    for split in ['train', 'test']:
        with open(os.path.join(CUSTOM_DATA_FOLDER, f'{split}_{size}.jsonl'), mode='w', encoding='utf8') as file:
            for line in tqdm.tqdm(danish_legal_dataset[split]['text']):
                if len(line) > 0 and not line.isspace():
                    ws_tokens = line.split(' ')
                    prev_idx = 0
                    for idx in range(max_sw_seq_length, len(ws_tokens) + max_sw_seq_length, max_sw_seq_length):
                        file.write(json.dumps({'text': ' '.join(ws_tokens[prev_idx:idx]) + '\n'}))
                        prev_idx = idx

