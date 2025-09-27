#!/bin/bash

task_name="text_code_gen"
model_name="mamba-hf"
nickname="130m"
lang="java"

if [ $lang == "java" ]
then
  model_name_or_path="../huggingface/state-spaces/mamba-130m-hf"
  train_data_file="../dataset/concode/train.json"
  eval_data_file="../dataset/concode/dev.json"
  test_data_file="../dataset/concode/test.json"
else
  echo "Invalid Programming Language for Text to Code Generation."
fi


CUDA_VISIBLE_DEVICES=0 python mamba_hf_main.py \
    --lang $lang \
    --output_dir ../saved_models/$task_name/$model_name/$lang/$nickname/lr8e-5 \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --is_mamba \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --visible_devices 0 \
    --num_train_epochs 5 \
    --block_size 512 \
    --train_batch_size 6 \
    --eval_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --learning_rate 8e-5 \
    --max_grad_norm 1.0 \
    --seed 42 \



#CUDA_VISIBLE_DEVICES=0 python mamba_hf_main.py \
#    --lang $lang \
#    --output_dir ../saved_models/$task_name/$model_name/$lang/$nickname/lr8e-5 \
#    --model_type $model_name \
#    --model_name_or_path $model_name_or_path \
#    --do_test \
#    --test_data_file $test_data_file \
#    --visible_devices 0 \
#    --num_train_epochs 5 \
#    --block_size 512 \
#    --max_target_length 100 \
#    --train_batch_size 6 \
#    --test_batch_size 1 \
#    --gradient_accumulation_steps 2 \
#    --learning_rate 8e-5 \
#    --max_grad_norm 1.0 \
#    --seed 42 \

