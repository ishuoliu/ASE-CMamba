## Can Mamba Be Better? An Experimental Evaluation of Mamba in Code Intelligence

### Pre-training Dataset

- The further pre-training dataset we used is available here: [pre-training dataset](https://huggingface.co/datasets/code-search-net/code_search_net).

### Downstream Dataset

- Line-level code completion: [dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line).
- Code generation: [dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code).
- Code clone detection: [dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench).

### Dependencies

```cmd
mamba-ssm == 2.2.3
causal_conv1d == 1.4.0
transformers == 4.48.0
peft == 0.14.0
python == 3.10
tree-sitter == 0.21.0
```

### Quick Start

- We release the further pre-trained Mamba models in `./pretrained_mamba` directory. You can also pre-train them usinig the scripts in `./pretrain` directory.
- We provide the output results of JavaCorpus in `./examples/JavaCorpus` directory.
- We provide a prompt example usded in the code clone detection task in `./examples/prompt_example.txt`. 
- Download the pre-training dataset and downstream datasets, put them in `./dataset` directory. 
- To quick start downstream tasks, take Line-level Code Completion (LCC) as an example.

#### Fine-tune

```shell
cd line_comple
python mamba_main.py \
    --lang java \
    --output_dir ../saved_models/line_comple/mamba/java/checkpoints \
    --model_type mamba \
    --model_name_or_path state-spaces/mamba-130m \
    --do_train \
    --train_data_file ../dataset/javaCorpus/token_completion/train.txt \
    --eval_data_file ../dataset/javaCorpus/token_completion/dev.txt \
    --lit_file ../dataset/javaCorpus/literals.json \
    --visible_devices 2 \
    --num_train_epochs 5 \
    --block_size 1024 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 42 
```

#### Evaluate

```shell
cd line_comple
python mamba_main.py \
    --lang java \
    --output_dir ../saved_models/line_comple/mamba/java/checkpoints \
    --model_type mamba \
    --model_name_or_path state-spaces/mamba-130m \
    --do_test \
    --test_data_file ../dataset/javaCorpus/token_completion/test.json \
    --lit_file ../dataset/javaCorpus/literals.json \
    --visible_devices 2 \
    --num_train_epochs 5 \
    --block_size 1024 \
    --max_target_length 100 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 42 
```
