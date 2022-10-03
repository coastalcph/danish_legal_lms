from tokenizers import models, normalizers, pre_tokenizers, decoders, processors, trainers
from tokenizers import Tokenizer
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import os
from data import DATA_DIR

CUSTOM_TOK_FOLDER = os.path.join(DATA_DIR, 'plms/danish-legal-lm-base')


def main(vocab_size=32000):

    # configure tokenizer
    backend_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    backend_tokenizer.normalizer = normalizers.Lowercase()
    backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    backend_tokenizer.decoder = decoders.ByteLevel()
    backend_tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 1),
                                                                    add_prefix_space=True, trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True
    )

    danish_dataset = load_dataset('DDSC/partial-danish-gigaword-no-twitter', split='train')
    danish_eurlex_dataset = load_dataset('multi_eurlex', 'da', split='train')
    danish_law_dataset = danish_dataset.filter(
        lambda example: example['source'] in ['retsinformationdk', 'retspraksis'])

    danish_legal_dataset = concatenate_datasets([danish_law_dataset, danish_eurlex_dataset])
    danish_legal_dataset = danish_legal_dataset.shuffle(seed=42)

    danish_legal_dataset = danish_legal_dataset.train_test_split(test_size=0.1, seed=42)

    # interleave datasets with sampling rates
    dataset = danish_legal_dataset['train']

    # train tokenizer
    backend_tokenizer.train_from_iterator(trainer=trainer, iterator=dataset['text'])

    # save tokenizer
    new_roberta_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        model_max_length=512,
        # padding_side="Set me if you want",
        # truncation_side="Set me if you want",
        # model_input_names="Set me if you want",
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        sep_token='</s>',
        pad_token='<pad>',
        cls_token='<s>',
        mask_token='<mask>',
    )

    new_roberta_tokenizer.save_pretrained(CUSTOM_TOK_FOLDER)
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_FOLDER)

    print(f'Trained BPE tokenizer with  a vocabulary of {vocab_size} sub-words successfully!')

    test_samples = dataset.select(range(500))
    for example in test_samples:
        print(example['text'][:500])
        print('-' * 150)
        print(tokenizer.tokenize(example['text'][:500]))
        print('-' * 150)

if __name__ == "__main__":
    main()
