#!/bin/bash

task_name="line_comple"
model_name="mamba2-hf"
nickname="130m"
lang="java"


if [ $lang == "java" ]
then
  model_name_or_path="../huggingface/state-spaces/mamba-130m-hf"
  train_data_file="../dataset/javaCorpus/token_completion/train.txt"
  eval_data_file="../dataset/javaCorpus/token_completion/dev.txt"
  test_data_file="../dataset/javaCorpus/token_completion/test.json"
  lit_file="../dataset/javaCorpus/literals.json"
elif [ $lang == "python" ]
then
  model_name_or_path="../huggingface/state-spaces/mamba-130m-hf"
  train_data_file="../dataset/py150/token_completion/train.txt"
  eval_data_file="../dataset/py150/token_completion/dev.txt"
  test_data_file="../dataset/py150/token_completion/test.json"
  lit_file="../dataset/py150/literals.json"
else
  echo "Invalid Programming Language for Line Completion."
fi


# Training
CUDA_VISIBLE_DEVICES=2 python mamba_hf_main.py \
    --lang $lang \
    --output_dir ../saved_models/$task_name/$model_name/$lang/$nickname/pretrain/lr5e-5 \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --lit_file $lit_file \
    --visible_devices 2 \
    --num_train_epochs 1 \
    --block_size 1024 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --use_pretrain \
    --checkpoint_prefix='checkpoint-best-loss-2/model.bin' \
    --pretrain_dir="../pretrained_mamba/mamba-hf/java/130m-scratch/lr5e-5" \



## Evaluating Line
#CUDA_VISIBLE_DEVICES=2 python mamba_hf_main.py \
#    --lang $lang \
#    --output_dir ../saved_models/$task_name/$model_name/$lang/$nickname/pretrain/lr5e-5 \
#    --model_type $model_name \
#    --model_name_or_path $model_name_or_path \
#    --do_test \
#    --test_data_file $test_data_file \
#    --lit_file $lit_file \
#    --visible_devices 2 \
#    --num_train_epochs 1 \
#    --block_size 1024 \
#    --max_target_length 100 \
#    --train_batch_size 4 \
#    --eval_batch_size 1 \
#    --gradient_accumulation_steps 4 \
#    --learning_rate 5e-5 \
#    --max_grad_norm 1.0 \
#    --seed 42 \


