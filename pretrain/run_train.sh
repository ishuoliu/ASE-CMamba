#!/bin/bash

model_name="mamba2"
nickname="370m"
lang="java"

model_name_or_path="../huggingface/state-spaces/mamba2-370m"


# Training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --lang $lang \
    --output_dir ../pretrained_mamba/$model_name/$lang/$nickname/lr5e-5 \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --visible_devices 0 \
    --epochs 1 \
    --max_length 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 42


