#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --model_name_or_path wanhin/msim-mt5-atien  \
    --tokenizer_name wanhin/msim-mt5-atien \
    --train_file data/luat-contrastive.csv \
    --output_dir checkpoints/msim-mt5-luat \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --metric_for_best_model stsb_spearman \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --load_best_model_at_end False\
    --save_total_limit 1 \
    --report_to 'wandb' \
    --logging_dir ./logs \
    --logging_steps 16 \
    --pooler_type 'avg' \
    --mlp_only_train False \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 False \
