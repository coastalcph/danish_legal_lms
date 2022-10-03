from transformers import AutoModel
from data import AUTH_KEY

results = ''
for model_name in ['Maltehb/danish-bert-botxo', 'Maltehb/aelaectra-danish-electra-small-uncased',
                   'coastalcph/danish-legal-lm-base', 'xlm-roberta-base']:
    model = AutoModel.from_pretrained(model_name, use_auth_token=AUTH_KEY)
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_total_params = model_total_params / 1e6
    results += f'Model "{model_name:>30} has {model_total_params:.1f}M number of parameters, ' \
               f'{model.config.vocab_size} vocabulary size, ' \
               f'{model.config.num_hidden_layers} layers, ' \
               f'{model.config.hidden_size} hidden units, and ' \
               f'{model.config.num_attention_heads} attention heads.\n'
    results += '-' * 150 + '\n'

print(results)
