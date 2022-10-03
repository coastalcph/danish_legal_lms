# Danish Legal Language Models

## Available Danish Language Models

| Model Name                                       | Layers    | Hidden Units | Attention Heads | Parameters |
|--------------------------------------------------|-----------|--------------|-----------------|------------|
| `Maltehb/danish-bert-botxo`                      | 12        | 768          | 12              | 110M       |
| `Maltehb/aelaectra-danish-electra-small-uncased` | 12        | 256          | 4               | -          |
| `coastalcph/danish-legal-lm-base`                | 12        | 768          | 12              | 110M       |
| `xlm-roberta-base`                               | 12        | 768          | 12              | 135M       |


## Danish Legal Data
This model is pre-trained on a combination of the Danish part of the MultiEURLEX (Chalkidis et al., 2021) dataset comprising EU legislation and two subsets (`retsinformationdk`, `retspraksis`) of the Danish Gigaword Corpus (Derczynski et al., 2021) comprising legal proceedings. It achieves the following results on the evaluation set.


## Evaluation

| Model Name                        | EURLEX | 
|-----------------------------------|--------|
| `Maltehb/danish-bert-botxo`       | -      | 
| `coastalcph/danish-legal-lm-base` | -      | 



