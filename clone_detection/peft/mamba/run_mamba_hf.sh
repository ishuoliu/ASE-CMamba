#!/bin/bash

task_name="clone_detection_peft"
model_name="mamba2-hf" 
nickname="130m"
lang="java"
peft_name="lora" 

model_name_or_path="../huggingface/state-spaces/mamba2-130m-hf"

data_dir="../dataset/bcb"
code_snippets_path="$data_dir/data.jsonl"
train_data_file="$data_dir/train.txt"
eval_data_file="$data_dir/valid.txt"
test_data_file="$data_dir/test.txt"

output_dir="./saved_models/$task_name/$peft_name/$model_name/$lang/$nickname"

echo "--- Running Training ---"
CUDA_VISIBLE_DEVICES=0 python mamba_hf_main.py \
    --output_dir $output_dir \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --peft_name $peft_name \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --code_snippets_path $code_snippets_path \
    --visible_devices 0 \
    --num_train_epochs 3 \
    --block_size 1024 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 43

echo "--- Running Testing ---"
CUDA_VISIBLE_DEVICES=0 python mamba_hf_main.py \
    --output_dir $output_dir \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --peft_name $peft_name \
    --do_test \
    --test_data_file $test_data_file \
    --code_snippets_path $code_snippets_path \
    --visible_devices 0 \
    --block_size 1024 \
    --eval_batch_size 12 \
    --seed 43