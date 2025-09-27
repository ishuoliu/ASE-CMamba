#!/bin/bash

task_name="line_comple"
model_name="CodeGPT"
nickname="small"
lang="java"

if [ $lang == "java" ]
then
  model_name_or_path="../huggingface/microsoft/CodeGPT-small-java"
  train_data_file="../dataset/javaCorpus/token_completion/train.txt"
  eval_data_file="../dataset/javaCorpus/token_completion/dev.txt"
  test_data_file="../dataset/javaCorpus/token_completion/test.json"
  lit_file="../dataset/javaCorpus/literals.json"
elif [ $lang == "python" ]
then
  model_name_or_path="../huggingface/microsoft/CodeGPT-small-py"
  train_data_file="../dataset/py150/token_completion/train.txt"
  eval_data_file="../dataset/py150/token_completion/dev.txt"
  test_data_file="../dataset/py150/token_completion/test.json"
  lit_file="../dataset/py150/literals.json"
else
  echo "Invalid Programming Language for Line Completion."
fi


# Training
CUDA_VISIBLE_DEVICES=2 python gpt_main.py \
    --lang $lang \
    --output_dir ../saved_models/$task_name/$model_name/$lang/$nickname/lr8e-5 \
    --model_type $model_name \
    --model_name_or_path $model_name_or_path  \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --lit_file $lit_file \
    --visible_devices 2 \
    --num_train_epochs 5 \
    --block_size 1024 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 8e-5 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --use_pretrain \
    --checkpoint_prefix='checkpoint-best-loss-2/model.bin' \
    --pretrain_dir="../pretrained_mamba/CodeGPT/java/scratch/lr5e-5" \


## Evaluating Line
#CUDA_VISIBLE_DEVICES=3 python gpt_main.py \
#    --lang $lang \
#    --output_dir ../saved_models/$task_name/$model_name/$lang/$nickname/lr2e-3 \
#    --model_type $model_name \
#    --model_name_or_path $model_name_or_path  \
#    --do_test \
#    --test_data_file $test_data_file \
#    --lit_file $lit_file \
#    --visible_devices 3 \
#    --num_train_epochs 5 \
#    --block_size 1024 \
#    --max_target_length 100 \
#    --train_batch_size 4 \
#    --eval_batch_size 1 \
#    --gradient_accumulation_steps 4 \
#    --learning_rate 2e-3 \
#    --max_grad_norm 1.0 \
#    --seed 42




