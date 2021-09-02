#! /bin/bash 
PYTHONPATH=. python scripts/train_lm.py \
--model_name xlm-roberta-base \
--train_folder $DATA_FOLDER \
--do_train \
--output_dir $DATA_OUTPUT \
--max_len 512 \
--save_strategy steps \
--save_steps 50000 \
--no_cuda \
--seed 42 \
--dataloader_num_workers 16 \
--logging_strategy steps \
--logging_steps 10 \
--per_device_train_batch_size 8000 \
--overwrite_output_dir \
--max_steps 50000 \
--learning_rate 4e-4 \
--warmup_steps 30000 \
