import json
import os.path
import numpy as np
from data import DATA_DIR
from data.the_legal_pile_original.compute_sampling_ratio import compute_sampling_rates

LEXGLUE_SUBSETS = ['ecthr_a', 'scotus', 'eurlex', 'ildc', 'case_hold', 'ledgar', 'contractnli_a']
PILE_SUBSETS = list(compute_sampling_rates()[0].keys())


MODELS = ['lexlms/roberta-tiny', 'lexlms/roberta-small', 'lexlms/roberta-base',
          'lexlms/roberta-large', 'nlpaueb/legal-bert-base-uncased']

# LEXGLUE
METRICS = {}
print('-' * 250)
print('LEXGLUE BENCHMARK')
for MODEL in MODELS:
    with open(os.path.join(DATA_DIR, 'PLMs', f'{MODEL}-mlm-eval', 'all_results.json')) as file:
        METRICS[MODEL] = json.load(file)
print('-' * 250)
line = f'{"TASK":>25}'
for MODEL in MODELS:
    line += F'{MODEL.upper():>25}\t\t'
print(line)
print('-' * 250)
for subset_name in LEXGLUE_SUBSETS:
    line = f'{subset_name.upper():>25}\t'
    for MODEL in MODELS:
        line += f'LOSS: {METRICS[MODEL][f"{subset_name}_eval_loss"]:.2f}\t' \
               f'ACCURACY: {METRICS[MODEL][f"{subset_name}_eval_accuracy"] * 100:.1f}\t\t'
    print(line)

print('-' * 250)
line = f'{"AVERAGED":>25}\t'
for MODEL in MODELS:
    avg_loss = np.mean([METRICS[MODEL][f"{value}_eval_loss"] for value in LEXGLUE_SUBSETS])
    avg_acc = np.mean([METRICS[MODEL][f"{value}_eval_accuracy"] for value in LEXGLUE_SUBSETS])
    line += f'LOSS: {avg_loss:.2f}\tACCURACY: {avg_acc * 100:.1f}\t\t'
print(line)
print('-' * 250)
print('\n')

# THE LEGAL PILE
METRICS = {}
print('-' * 250)
print('THE LEGAL PILE CORPUS')
for MODEL in MODELS:
    with open(os.path.join(DATA_DIR, 'PLMs', f'{MODEL}-pile-mlm-eval', 'all_results.json')) as file:
        METRICS[MODEL] = json.load(file)
print('-' * 250)
line = f'{"TASK":>25}'
for MODEL in MODELS:
    line += F'{MODEL.upper():>25}\t\t'
print(line)
print('-' * 250)
for subset_name in PILE_SUBSETS:
    line = f'{subset_name.upper():>25}\t'
    for MODEL in MODELS:
        line += f'LOSS: {METRICS[MODEL][f"{subset_name}_eval_loss"]:.2f}\t' \
               f'ACCURACY: {METRICS[MODEL][f"{subset_name}_eval_accuracy"] * 100:.1f}\t\t'
    print(line)

print('-' * 250)
line = f'{"AVERAGED":>25}\t'
for MODEL in MODELS:
    avg_loss = np.mean([METRICS[MODEL][f"{value}_eval_loss"] for value in PILE_SUBSETS])
    avg_acc = np.mean([METRICS[MODEL][f"{value}_eval_accuracy"] for value in PILE_SUBSETS])
    line += f'LOSS: {avg_loss:.2f}\tACCURACY: {avg_acc * 100:.1f}\t\t'
print(line)
print('-' * 250)



