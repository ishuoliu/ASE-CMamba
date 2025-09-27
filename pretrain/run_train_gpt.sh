#!/bin/bash

model_name="CodeGPT"
nickname="scratch"
lang="java"

model_name_or_path="../huggingface/microsoft/CodeGPT-small-java"


# Training
CUDA_VISIBLE_DEVICES=0 python train_gpt.py \
    --lang $lang \
    --output_dir ../pretrained_mamba/$model_name/$lang/$nickname/lr5e-5 \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --visible_devices 0 \
    --epochs 3 \
    --max_length 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 42
