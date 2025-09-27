#!/bin/bash

model_name="CodeGPT"
nickname="small"
lang="java"
task="clone_detection"

model_name_or_path="../huggingface/microsoft/CodeGPT-small-java" #

data_dir="../dataset/bcb"
code_snippets_path="$data_dir/data.jsonl"
train_data_file="$data_dir/sampled_train.txt"
eval_data_file="$data_dir/sampled_valid.txt"
test_data_file="$data_dir/sampled_test.txt"

output_dir="./saved_models/$model_name/$lang/$nickname/$task/lr5e-5"

echo "--- Running Training ---"
CUDA_VISIBLE_DEVICES=0 python gpt_main.py \
    --output_dir $output_dir \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --code_snippets_path $code_snippets_path \
    --visible_devices 0 \
    --num_train_epochs 5 \
    --block_size 1024 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 46 \
    --train_data_ratio 1.0
echo "--- Running Testing ---"
CUDA_VISIBLE_DEVICES=0 python gpt_main.py \
    --output_dir $output_dir \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --do_test \
    --test_data_file $test_data_file \
    --code_snippets_path $code_snippets_path \
    --visible_devices 0 \
    --block_size 1024 \
    --eval_batch_size 12 \
    --seed 46