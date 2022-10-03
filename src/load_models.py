from transformers import AutoModel
from data import AUTH_KEY

results = ''
for model_name in ['Maltehb/danish-bert-botxo', 'Maltehb/aelaectra-danish-electra-small-uncased',
                   'coastalcph/danish-legal-lm-base', 'xlm-roberta-base']:
    model = AutoModel.from_pretrained(model_name, use_auth_token=AUTH_KEY)
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_total_params = model_total_params / 1e6
    model_name = "\'" + model_name + "\'"
    results += f'Model {model_name:>50} has {model_total_params:.1f}M number of parameters, ' \
               f'{model.config.vocab_size} vocabulary size, ' \
               f'{model.config.num_hidden_layers} layers, ' \
               f'{model.config.hidden_size} hidden units, and ' \
               f'{model.config.num_attention_heads} attention heads.\n'
    results += '-' * 150 + '\n'

print(results)

#
# Model                        'Maltehb/danish-bert-botxo' has 110.6M number of parameters, 32000 vocabulary size, 12 layers, 768 hidden units, and 12 attention heads.
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Model   'Maltehb/aelaectra-danish-electra-small-uncased' has 13.7M number of parameters, 32000 vocabulary size, 12 layers, 256 hidden units, and 4 attention heads.
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Model                  'coastalcph/danish-legal-lm-base' has 110.6M number of parameters, 32000 vocabulary size, 12 layers, 768 hidden units, and 12 attention heads.
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Model                                 'xlm-roberta-base' has 278.0M number of parameters, 250002 vocabulary size, 12 layers, 768 hidden units, and 12 attention heads.
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#

