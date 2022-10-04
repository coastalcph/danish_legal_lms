# Danish Legal Language Models

## Available Danish Language Models

| Model Name                                       | Layers / Units /  Heads | Vocab. | Parameters |
|--------------------------------------------------|-------------------------|--------|------------|
| `Maltehb/danish-bert-botxo`                      | 12 / 768 / 12           | 32K    | 110M       |
| `Maltehb/aelaectra-danish-electra-small-uncased` | 12 / 256 / 4            | 32K    | 14M        |
| `coastalcph/danish-legal-lm-base`                | 12 / 768 / 12           | 32K    | 110M       |
| `coastalcph/danish-legal-bert-base`              | 12 / 768 / 12           | 32K    | 110M       |
| `coastalcph/danish-legal-xlm-base`               | 12 / 768 / 12           | 32K    | 110M       |
| `xlm-roberta-base`                               | 12 / 768 / 12           | 256K   | 278M       |


## Danish Legal Data
This model is pre-trained on a combination of the Danish part of the MultiEURLEX (Chalkidis et al., 2021) dataset comprising EU legislation and two subsets (`retsinformationdk`, `retspraksis`) of the Danish Gigaword Corpus (Derczynski et al., 2021) comprising legal proceedings. It achieves the following results on the evaluation set.


## Evaluation

| Model Name                          | EURLEX Val. | EURLEX Test | 
|-------------------------------------|-------------|-------------|
| `Maltehb/danish-bert-botxo`         | 73.7 / 42.8 | 67.6 / 38.2 | 
| `coastalcph/danish-legal-lm-base`   | 75.1 / 46.5 | 69.1 / 41.9 | 
| `coastalcph/danish-legal-bert-base` | TBA         | TBA         | 
| `coastalcph/danish-legal-xlm-base`  | TBA         | TBA         | 



