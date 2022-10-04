import os
import torch
from torch.nn import Parameter
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from data import DATA_DIR
from flota_tokenizer import FlotaTokenizer
import numpy as np
import unidecode
import copy
import argparse


def warm_start_model():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--teacher_model_name', default='xlm-roberta-base')
    parser.add_argument('--teacher_start', default='▁')
    parser.add_argument('--student_model_name', default='../data/plms/danish-legal-xlm-base')
    parser.add_argument('--student_start', default='Ġ')
    parser.add_argument('--use_flota', default=True)
    parser.add_argument('--flota_mode', default='longest', choices=['flota', 'longest', 'first'])
    parser.add_argument('--auth_token', default=None)
    config = parser.parse_args()

    # load tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name,
                                                      use_auth_token=config.auth_token)
    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_name,
                                                      use_auth_token=config.auth_token)
    # use FLOTA tokenizer of Hofmann et al. (2022) to decrease over-fragmentation
    if config.use_flota:
        teacher_flota_tokenizer = FlotaTokenizer(config.teacher_model_name,
                                                 mode=config.flota_mode,
                                                 k=1 if config.flota_mode in ['longest', 'first'] else 4)

    # define student-teacher mappings
    student_teacher_mapping_ids = {}
    student_teacher_mapping_tokens = {}
    student_teacher_mapping_compound_tokens = {}

    # Build teacher vocab dict
    teacher_vocab = [(token_id, token) for token, token_id in teacher_tokenizer.vocab.items()]
    teacher_vocab.sort()
    teacher_vocab = {token: token_id for (token_id, token) in teacher_vocab}
    teacher_vocab_lowercased = {token.lower(): token_id for (token, token_id) in teacher_vocab.items()}
    teacher_vocab_ids = {token_id: token for token, token_id in teacher_vocab.items()}
    TEACHER_START = config.teacher_start

    # Build student vocab dict
    student_vocab = [(token_id, token) for token, token_id in student_tokenizer.vocab.items()]
    student_vocab.sort()
    student_vocab = {token: token_id for (token_id, token) in student_vocab}
    STUDENT_START =config.student_start

    # statistics counter
    identical_tokens = 0
    semi_identical_tokens = 0
    semi_identical_normalized_tokens = 0
    longest_first_tokens = 0
    compound_tokens = 0
    unk_tokens = 0
    flota_diffs = []

    # look up for student tokens in teacher's vocabulary
    for original_token, token_id in student_vocab.items():
        token = copy.deepcopy(original_token).replace(STUDENT_START, TEACHER_START)
        # perfect match (e.g., "document" --> "document")
        if token in teacher_vocab:
            student_teacher_mapping_ids[token_id] = teacher_vocab[token]
            student_teacher_mapping_tokens[token] = token
            identical_tokens += 1
        # perfect match with cased version of token (e.g., "paris" --> "Paris")
        elif token.lower() in teacher_vocab_lowercased:
            student_teacher_mapping_ids[token_id] = teacher_vocab_lowercased[token.lower()]
            student_teacher_mapping_tokens[token] = teacher_vocab_ids[teacher_vocab_lowercased[token.lower()]]
            semi_identical_tokens += 1
        # match with non-starting token (e.g., "concat" --> "_concat")
        elif token.replace(TEACHER_START, '') in teacher_vocab:
            student_teacher_mapping_ids[token_id] = teacher_vocab[token.replace(TEACHER_START, '')]
            student_teacher_mapping_tokens[token] = token.replace(TEACHER_START, '')
            semi_identical_tokens += 1
        # match with cased version of non-starting token (e.g., "ema" --> "_EMA")
        elif token.lower().replace(TEACHER_START, '') in teacher_vocab_lowercased:
            student_teacher_mapping_ids[token_id] = teacher_vocab_lowercased[token.lower().replace(TEACHER_START, '')]
            student_teacher_mapping_tokens[token] = teacher_vocab_ids[teacher_vocab_lowercased[token.lower().replace(TEACHER_START, '')]]
            semi_identical_normalized_tokens += 1
        # normalized version of token in vocab -> map to the normalized version of token
        # (e.g., 'garçon' --> 'garcon')
        elif unidecode.unidecode(token) in teacher_vocab:
            student_teacher_mapping_ids[token_id] = teacher_vocab[unidecode.unidecode(token)]
            student_teacher_mapping_tokens[token] = unidecode.unidecode(unidecode.unidecode(token))
            semi_identical_normalized_tokens += 1
        # normalized version of uncased token in vocab -> map to the normalized version of token
        # (e.g., 'Garçon' --> 'garçon' --> 'garcon')
        elif unidecode.unidecode(token).lower() in teacher_vocab_lowercased:
            student_teacher_mapping_ids[token_id] = teacher_vocab_lowercased[unidecode.unidecode(token).lower()]
            student_teacher_mapping_tokens[token] = unidecode.unidecode(unidecode.unidecode(token).lower())
            semi_identical_normalized_tokens += 1
        else:
            # tokenize token (e.g., "unprecedented" --> ['_un', 'prec', 'edent', 'ed'])
            token = token.replace(STUDENT_START, ' ').replace(TEACHER_START, ' ')
            sub_words = teacher_tokenizer.encode(' ' + token, add_special_tokens=False)
            sub_words_tokens = [teacher_vocab_ids[sub_word] for sub_word in sub_words]
            if config.use_flota:
                # tokenize token with FLOTA (e.g., "unprecedented" --> ['_un', 'precedented'])
                flota_sub_words = teacher_flota_tokenizer.encode(token)[:-2]
                flota_sub_words_tokens = [teacher_vocab_ids[sub_word] for sub_word in flota_sub_words]
            # keep the list with the fewer sub-words
            if config.use_flota and len(flota_sub_words_tokens) and len(flota_sub_words_tokens) <= len(sub_words_tokens):
                flota_diffs.append((sub_words_tokens, flota_sub_words_tokens))
                sub_words = flota_sub_words
                sub_words_tokens = flota_sub_words_tokens
            # sub-word token -> map to the sub-word (e.g., "_μ" --> ["μ"] --> "μ")
            if len(sub_words) == 1 and sub_words[0] != teacher_tokenizer.unk_token_id:
                student_teacher_mapping_ids[token_id] = sub_words[0]
                student_teacher_mapping_tokens[token] = sub_words_tokens[0]
                if sub_words_tokens[0].replace(STUDENT_START, '').replace(TEACHER_START, '') == \
                        original_token.replace(STUDENT_START, '').replace(TEACHER_START, ''):
                    semi_identical_tokens += 1
                else:
                    longest_first_tokens += 1
            # list of sub-words w/o <unk> -> map to the list (e.g., 'overqualified' --> ['over', '_qualified'] )
            elif len(sub_words) >= 2 and teacher_tokenizer.unk_token_id not in sub_words:
                student_teacher_mapping_ids[token_id] = sub_words
                student_teacher_mapping_tokens[token] = sub_words_tokens
                student_teacher_mapping_compound_tokens[token] = sub_words_tokens
                compound_tokens += 1
            else:
                # list of sub-words w/ <unk> -> map to the list (e.g., 'Ω-power' --> [<unk>, '-power'] --> '-power')
                if len(sub_words) > 1 and set(sub_words) != {teacher_tokenizer.unk_token_id}:
                    student_teacher_mapping_compound_tokens[token] = sub_words.remove(teacher_tokenizer.unk_token_id)
                    compound_tokens += 1
                # <unk> -> map to <unk>
                else:
                    # No hope use <unk> (e.g., '晚上好' --> <unk>)
                    student_teacher_mapping_tokens[token] = '<unk>'
                    print(f'Token "{token}" not in vocabulary, replaced with UNK.')
                    unk_tokens += 1

    # print mapping statistics
    print(f'The student-teacher mapping algorithm led to:')
    print(f'- ({str(identical_tokens):>5}) ({identical_tokens/len(student_vocab)*100:.1f}%) identical tokens ')
    print(f'- ({str(semi_identical_normalized_tokens):>5}) ({semi_identical_normalized_tokens/len(student_vocab)*100:.1f}%) semi-identical normalized tokens.')
    print(f'- ({str(semi_identical_tokens):>5}) ({semi_identical_tokens/len(student_vocab)*100:.1f}%) semi-identical tokens.')
    print(f'- ({str(longest_first_tokens):>5}) ({longest_first_tokens/len(student_vocab)*100:.1f}%) semi-identical tokens.')
    print(f'- ({str(compound_tokens):>5}) ({compound_tokens/len(student_vocab)*100:.1f}%) compound tokens.')
    print(f'- ({str(unk_tokens):>5}) ({unk_tokens/len(student_vocab)*100:.1f}%) unknown tokens.')
    if config.use_flota:
        avg_flota_chunks = sum([len(tokens[1]) for tokens in flota_diffs]) / len(flota_diffs)
        avg_standard_chunks = sum([len(tokens[0]) for tokens in flota_diffs]) / len(flota_diffs)
        flota_diff = sum([1 for tokens in flota_diffs if len(tokens[1]) < len(tokens[0])])
        print(f'FLOTA: Decreased fragmentation for {str(flota_diff):>5} ({flota_diff/len(flota_diffs)*100:.1f}%) tokens '
              f'with an average of {avg_standard_chunks - avg_flota_chunks:.1f} sub-words.')

    # load dummy student model
    student_model_config = AutoConfig.from_pretrained(config.student_model_name)
    roberta_model = AutoModelForMaskedLM.from_config(student_model_config)

    # load teacher model
    xlm_roberta_model = AutoModelForMaskedLM.from_pretrained(config.teacher_model_name)

    # copy positional and token type embeddings
    roberta_model.roberta.embeddings.position_embeddings.\
        load_state_dict(xlm_roberta_model.roberta.embeddings.position_embeddings.state_dict())
    roberta_model.roberta.embeddings.token_type_embeddings.\
        load_state_dict(xlm_roberta_model.roberta.embeddings.token_type_embeddings.state_dict())
    roberta_model.roberta.embeddings.LayerNorm.\
        load_state_dict(xlm_roberta_model.roberta.embeddings.LayerNorm.state_dict())

    # Extract teacher word embeddings
    word_embeddings_matrix = copy.deepcopy(xlm_roberta_model.roberta.embeddings.word_embeddings.weight.detach())
    word_embeddings = [word_embeddings_matrix[teacher_id] if isinstance(teacher_id, int)
                       else word_embeddings_matrix[teacher_id].mean(dim=0)
                       for student_id, teacher_id in student_teacher_mapping_ids.items()]
    word_embeddings = torch.concat(word_embeddings, dim=0)

    # replace student's word embeddings matrix
    roberta_model.roberta.embeddings.word_embeddings.weight = Parameter(word_embeddings)

    # Copy transformer block
    roberta_model.roberta.encoder.load_state_dict(xlm_roberta_model.roberta.encoder.state_dict())

    # copy embeddings to lm_head
    roberta_model.lm_head.decoder.weight = Parameter(word_embeddings)

    # Extract teacher word embeddings, use the centroid for compound tokens
    lm_head_biases = copy.deepcopy(xlm_roberta_model.lm_head.bias.detach())
    lm_head_biases = [lm_head_biases[teacher_id] if isinstance(teacher_id, int)
                      else lm_head_biases[teacher_id].mean(dim=0)
                      for student_id, teacher_id in student_teacher_mapping_ids.items()]
    lm_head_biases = torch.as_tensor(np.array(lm_head_biases))

    # replace student's word embeddings matrix
    roberta_model.lm_head.decoder.bias = Parameter(lm_head_biases)

    # save frankenstein model
    roberta_model.save_pretrained(os.path.join(DATA_DIR, 'plms/danish-legal-xlm-base'))


if __name__ == '__main__':
    warm_start_model()
