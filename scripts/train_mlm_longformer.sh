export WANDB_PROJECT="danish-lex-lm"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES = 4,5,6,7

MODEL_MAX_LENGTH=2048
MODEL_PATH='plms/danish-legal-longformer-base'
BATCH_SIZE=16

python3 src/xla_spawn.py --num_cores=8 src/pretraining/train_mlm.py \
    --model_name_or_path data/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name danish_legal_pile \
    --output_dir data/${MODEL_PATH}-mlm \
    --overwrite_output_dir \
    --logging_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 32000 \
    --save_strategy steps \
    --save_steps 32000 \
    --save_total_limit 3 \
    --max_steps 64000 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --line_by_line \
    --max_eval_samples 12800
