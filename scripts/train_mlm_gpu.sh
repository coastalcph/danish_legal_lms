export WANDB_PROJECT="danish-lex-lm"
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_PATH='plms/danish-legal-lm-base'
BATCH_SIZE=16
ACCUMULATION_STEPS=2

python src/pretraining/train_mlm.py \
    --model_name_or_path data/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name danish_legal_pile \
    --output_dir data/${MODEL_PATH}-mlm \
    --overwrite_output_dir \
    --logging_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 5 \
    --max_steps 500000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --line_by_line \
    --max_eval_samples 10000 \
    --fp16 \
    --fp16_full_eval

