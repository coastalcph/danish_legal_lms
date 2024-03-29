export WANDB_PROJECT="danish-lm"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1

MODEL_MAX_LENGTH=1024
MODEL_PATH='coastalcph/danish-legal-longformer-base'
CONCEPT_LEVEL='level_2'
BATCH_SIZE=16

python src/finetuning/train_eurlex.py \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --do_eval \
    --do_predict \
    --concept_level ${CONCEPT_LEVEL} \
    --output_dir data/logs/${MODEL_PATH}/eurlex/${CONCEPT_LEVEL} \
    --overwrite_output_dir \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True

