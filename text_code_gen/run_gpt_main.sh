#!/bin/bash

model_name="CodeGPT"
nickname="small"
lang="java"

if [ $lang == "java" ]
then
  model_name_or_path="../huggingface/microsoft/CodeGPT-small-java"
  train_data_file="../dataset/concode/train.json"
  eval_data_file="../dataset/concode/dev.json"
  test_data_file="../dataset/concode/test.json"
else
  echo "Invalid Programming Language for Text to Code Generation."
fi


CUDA_VISIBLE_DEVICES=1 python gpt_main.py \
    --lang $lang \
    --output_dir ./saved_models/$model_name/$lang/$nickname/lr5e-5 \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --visible_devices 1 \
    --num_train_epochs 1 \
    --block_size 512 \
    --train_batch_size 6 \
    --eval_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 42

#
#CUDA_VISIBLE_DEVICES=1 python gpt_main.py \
#    --lang $lang \
#    --output_dir ./saved_models/$model_name/$lang/$nickname/lr5e-5 \
#    --model_type $model_name \
#    --model_name_or_path $model_name_or_path \
#    --do_test \
#    --test_data_file $test_data_file \
#    --visible_devices 1 \
#    --num_train_epochs 1 \
#    --block_size 512 \
#    --max_target_length 100 \
#    --train_batch_size 6 \
#    --test_batch_size 1 \
#    --gradient_accumulation_steps 2 \
#    --learning_rate 5e-5 \
#    --max_grad_norm 1.0 \
#    --seed 42
