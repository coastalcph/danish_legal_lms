# Preliminaries
Clone the repository into a local folder and let `DANISH_LM` be the folder containing the project. 

Run the following command to install the library necessary for the training.
```bash
cd $DANISH_LM
chmod +x scripts/config_env.sh
./scripts/config_env.sh
```

# Data
We assume that there is a folder `TRAINING_DATA` containing multiple `.txt` files storing the data for training.

# Training from Scratch

## Training Tokenizer
Run the following command to train a tokenizer tailored on the given data. There are a two optional parameters
that one can specify:
- `vocab_size`: defines the number of unique tokens within the vocabulary learnt from the file. Note that a 
token is not a proper word but may be a subword (default is 52000).
- `min_freq`: defines the minimum frequency of a token to be included within the vocabular (default is 2).

```bash
cd $DANISH_LM
mkdir danish_tokenizer

PYTHONPATH=. python scripts/train_tokenizer.py --data_folder $TRAINING_DATA --out_folder danish_tokenizer/ --vocab_size 52000 --min_freq 2
```
This will populate the `danish_tokenizer` folder with two files: `mewrges.txt` and `vocab.json`.

## Training the model
To train the model simply type:
```bash
export DATA_FOLDER=$TRAINING_DATA
export DATA_OUT=/path/to/output/folder
./scripts/train_lm.sh
```
The `DATA_FOLDER` and `DATA_OUT` folders are paths to the training and where the model will be saved, respectively.
One can change the parameters within the `train_lm.sh` file, especially those for `config_overrides`
in order to change the topology of the model (i.e., making it larger or smaller).

The command will train a Roberta large-like model by reading the data in `$DATA_FOLDER` and save the model
every `10000` steps in `$DATA_OUT`. The training can be monitored by typing
```bash
tensorboard dev upload --logdir $DATA_OUT/runs/
```
and follow the instructions.

# Training from a Pretrained Model
Rather than training from scratch, one can initialise the mode weights with those of a pretrained model, e.g.
`Multilingual BERT`, `XLM-Roberta`, etc.
To this end, there is no need to train a tokenizer and we can jump directly to continue the training of 
the model as follows:
```bash
export DATA_FOLDER=$TRAINING_DATA
export DATA_OUT=/path/to/output/folder
./scripts/train_lm_pretrained.sh
```
The `DATA_FOLDER` and `DATA_OUT` folders are paths to the training and where the model will be saved, respectively.
One can change the parameters within the `train_lm_pretrained.sh` file.
To select the pretrained weights one can change the value of `model_name` variable and choose among those
available in https://huggingface.co/transformers/pretrained_models.html
Relevant names of models are:
- `bert-base-multilingual-cased` a BERT model pretrained in 104 languages (including Danish).
- `xlm-roberta-base` a RoBERTa model pretrained in 100 languages (including Danish).
- `xlm-roberta-large` a larger (and usually better-performing) version of the model above.
- `Maltehb/danish-bert-botxo` a Danish version of BERT trained on CommonCrawl, Wikipedia, dindebat.dk, hestenettet.dk and OpenSubtitles. For more information see https://github.com/certainlyio/nordic_bert

The command will train the specified model by reading the data in `$DATA_FOLDER` and save the model
every `10000` steps in `$DATA_OUT`. The training can be monitored by typing
```bash
tensorboard dev upload --logdir $DATA_OUT/runs/
```
and follow the instructions.

