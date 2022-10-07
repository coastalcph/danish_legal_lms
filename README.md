# Danish Legal Language Models

## Available Language Models for Danish

| Model Name                                                                                                  | Layers / Units /  Heads | Vocab. | Parameters | Legal              |
|-------------------------------------------------------------------------------------------------------------|-------------------------|--------|------------|--------------------|
| [`Maltehb/danish-bert-botxo`](https://huggingface.co/Maltehb/danish-bert-botxo)                             | 12 / 768 / 12           | 32K    | 110M       | :x:                |
| [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base)                                               | 12 / 768 / 12           | 256K   | 278M       | :x:                |
| [`coastalcph/danish-legal-lm-base`](https://huggingface.co/coastalcph/danish-legal-lm-base)                 | 12 / 768 / 12           | 32K    | 110M       | :white_check_mark: |
| [`coastalcph/danish-legal-bert-base`](https://huggingface.co/coastalcph/danish-legal-bert-base)             | 12 / 768 / 12           | 32K    | 110M       | :white_check_mark: |
| [`coastalcph/danish-legal-longformer-base`](https://huggingface.co/coastalcph/danish-legal-longformer-base) | 12 / 768 / 12           | 32K    | 134M       | :white_check_mark: |
| [`coastalcph/danish-legal-xlm-base`](https://huggingface.co/coastalcph/danish-legal-xlm-base)               | 12 / 768 / 12           | 32K    | 110M       | :white_check_mark: |


## Danish Legal Pile
This model is pre-trained on a combination of the Danish part of the [MultiEURLEX](https://huggingface.co/datasets/multi_eurlex) (Chalkidis et al., 2021) dataset comprising 65k EU laws and two subsets (`retsinformationdk`, `retspraksis`) of the [Danish Gigaword Corpus](https://huggingface.co/datasets/DDSC/partial-danish-gigaword-no-twitter) (Derczynski et al., 2021) comprising legal proceedings. It achieves the following results on the evaluation set.

| Model Name                          | Loss | Accuracy | 
|-------------------------------------|------|----------|
| `Maltehb/danish-bert-botxo`         | 22.3 | 7.038    | 
| `coastalcph/danish-legal-lm-base`   | 84.8 | 0.651    | 
| `coastalcph/danish-legal-bert-base` | 80.1 | 0.878    | 
| `coastalcph/danish-legal-bert-base` | 82.5 | 0.768    | 
| `coastalcph/danish-legal-xlm-base`  | 83.1 | 0.727    | 


## Benchmarking

| Model Name                                | EURLEX Val. | EURLEX Test | 
|-------------------------------------------|-------------|-------------|
| `Maltehb/danish-bert-botxo`               | 73.7 / 42.8 | 67.6 / 38.2 | 
| `coastalcph/danish-legal-lm-base`         | 75.1 / 46.5 | 69.1 / 41.9 | 
| `coastalcph/danish-legal-bert-base`       | 75.0 / 50.4 | 68.9 / 44.3 | 
| `coastalcph/danish-legal-xlm-base`        | TBA         | TBA         | 
| `coastalcph/danish-legal-longformer-base` | 75.7 / 52.9 | 69.6 / 47.0 | 


The best model `coastalcph/danish-legal-longformer-base` is available on HuggingFace Hub (https://huggingface.co/coastalcph/danish-legal-longformer-eurlex) with instructions on how can be used as text classifier or efature extractor.

## Code Base

### Train new RoBERTa LM

```shell
sh train_mlm_gpu.sh
```

### Modify pre-trained XLM-R

```bash
export PYTHONPATH=.
python src/mod_teacher_model.py --teacher_model_path coastalcph/danish-legal-lm-base --student_model_path coastalcph/danish-legal-lm-base
```

### Longformerize pre-trained RoBERTa LM

```bash
export PYTHONPATH=.
python src/longformerize_model.py --roberta_model_path coastalcph/danish-legal-lm-base --max_length 2048 --attention_window 128
```