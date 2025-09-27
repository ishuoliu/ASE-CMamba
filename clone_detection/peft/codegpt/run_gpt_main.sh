#!/bin/bash

task_name="peft_clone_detection"
model_name="CodeGPT"
nickname="small"
lang="java"
peft_name="lora"

if [ $lang == "java" ]
then
  model_name_or_path="../huggingface/microsoft/CodeGPT-small-java"

  code_snippets_path="../dataset/bcb/data.jsonl"
  train_data_file="../dataset/bcb/sampled_train.txt"
  eval_data_file="../dataset/bcb/sampled_valid.txt"
  test_data_file="../dataset/bcb/sampled_test.txt"
else
  echo "Invalid Programming Language for Code Clone Detection."
  exit 1 
fi

CUDA_VISIBLE_DEVICES=0 python gpt_main.py \
    --lang $lang \
    --output_dir ../saved_models/$task_name/$model_name/$lang/$peft_name/$nickname/lr5e-5 \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --peft_name $peft_name \
    --do_train \
    --code_snippets_path $code_snippets_path \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --test_data_file $test_data_file \
    --visible_devices 0 \
    --num_train_epochs 5 \
    --block_size 1024 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 46 \
    --train_data_ratio 1.0 \
    --do_test