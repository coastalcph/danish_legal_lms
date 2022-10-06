export WANDB_PROJECT="danish-lex-lm"
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_PATH='coastalcph/danish-legal-lm-base'
BATCH_SIZE=32

python src/pretraining/eval_mlm.py \
    --model_name_or_path ${MODEL_PATH} \
    --do_eval \
    --dataset_name danish_legal_pile \
    --output_dir data/${MODEL_PATH}-eval \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --line_by_line
