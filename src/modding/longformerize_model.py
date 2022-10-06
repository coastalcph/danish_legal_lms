import argparse
import copy
import os.path
import warnings
from data import DATA_DIR, AUTH_KEY
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
warnings.filterwarnings("ignore")


def convert_roberta_to_htf():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--roberta_model_path', default='coastalcph/danish-legal-lm-base')
    parser.add_argument('--longformer_model_path', default=os.path.join(DATA_DIR, 'plms/danish-legal-longformer-base'))
    parser.add_argument('--max_length', default=2048, type=int)
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument('--auth_token', default=AUTH_KEY, type=str)
    config = parser.parse_args()

    # load pre-trained bert model and tokenizer
    roberta_model = AutoModelForMaskedLM.from_pretrained(config.roberta_model_path,
                                                         use_auth_token=config.auth_token)

    tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_path,
                                              use_auth_token=config.auth_token,
                                              model_max_length=config.max_length)

    # load dummy config and change specifications
    roberta_config = roberta_model.config
    lf_config = AutoConfig.from_pretrained(config.longformer_model_path)
    # Text length parameters
    lf_config.max_position_embeddings = config.max_length + roberta_config.pad_token_id + 2
    lf_config.model_max_length = config.max_length
    # Transformer parameters
    lf_config.attention_window = [config.window_size] * roberta_config.num_hidden_layers
    # Vocabulary parameters
    lf_config.vocab_size = roberta_config.vocab_size
    lf_config.pad_token_id = roberta_config.pad_token_id
    lf_config.bos_token_id = roberta_config.bos_token_id
    lf_config.eos_token_id = roberta_config.eos_token_id
    lf_config.cls_token_id = tokenizer.cls_token_id
    lf_config.sep_token_id = tokenizer.sep_token_id
    lf_config.type_vocab_size = roberta_config.type_vocab_size

    # load dummy longformer model
    lf_model = AutoModelForMaskedLM.from_config(lf_config)

    # copy embeddings
    k = 2
    step = roberta_config.max_position_embeddings - 2
    while k < lf_config.max_position_embeddings - 1:
        if k + step >= lf_config.max_position_embeddings:
            lf_model.longformer.embeddings.position_embeddings.weight.data[
            k:] = roberta_model.roberta.embeddings.position_embeddings.weight[
                  2:(roberta_config.max_position_embeddings + 2)]
        else:
            lf_model.longformer.embeddings.position_embeddings.weight.data[
            k:(k + step)] = roberta_model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    lf_model.longformer.embeddings.word_embeddings.load_state_dict(
        roberta_model.roberta.embeddings.word_embeddings.state_dict())
    lf_model.longformer.embeddings.token_type_embeddings.load_state_dict(
        roberta_model.roberta.embeddings.token_type_embeddings.state_dict())
    lf_model.longformer.embeddings.LayerNorm.load_state_dict(roberta_model.roberta.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    for i in range(len(roberta_model.roberta.encoder.layer)):
        # generic
        lf_model.longformer.encoder.layer[i].intermediate.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].intermediate.dense)
        lf_model.longformer.encoder.layer[i].output.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].output.dense)
        lf_model.longformer.encoder.layer[i].output.LayerNorm = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].output.LayerNorm)
        # attention output
        lf_model.longformer.encoder.layer[i].attention.output.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.output.dense)
        lf_model.longformer.encoder.layer[i].attention.output.LayerNorm = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.output.LayerNorm)
        # local q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.value)
        # global q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.value)

    # copy lm_head
    lf_model.lm_head.load_state_dict(roberta_model.lm_head.state_dict())

    # save model
    lf_model.save_pretrained(config.longformer_model_path)

    # save tokenizer
    tokenizer.save_pretrained(config.longformer_model_path)

    # re-load model
    lf_model = AutoModelForMaskedLM.from_pretrained(config.longformer_model_path)
    lf_tokenizer = AutoTokenizer.from_pretrained(config.longformer_model_path)
    print(f'Longformer model warm-started from "{config.roberta_model_path}" is ready to run!')


if __name__ == '__main__':
    convert_roberta_to_htf()
