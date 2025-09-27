import argparse
import random
import numpy as np
import torch
import os
import json
import sys

import transformers
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    AutoTokenizer, 
    MambaForCausalLM, 
    AutoModelForCausalLM,
    MambaConfig
)
from transformers import GPT2ForSequenceClassification
# ------------------------------------

sys.path.append("../")
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--train_data_file", default="../dataset/concode/train.json", type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--eval_data_file", default="../dataset/concode/dev.json", type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_data_file", default="../dataset/concode/test.json", type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--code_snippets_path", default=None, type=str,
                        help="The path to the jsonl file containing the code snippets for clone detection.")
    parser.add_argument("--model_type", default="CodeGPT", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--visible_devices", default="0", type=str,
                        help="The indices of visible GPUs")

    parser.add_argument("--max_target_length", default=100, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--is_mamba", action='store_true')
    parser.add_argument("--use_pretrain", action='store_true')
    parser.add_argument("--checkpoint_prefix", default=None, type=str)
    parser.add_argument("--pretrain_dir", default=None, type=str)

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for testing.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="beam size for beam search")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", type=int, default=0.1)
    parser.add_argument("--train_data_ratio", type=float, default=1.0,
                        help="The proportion of the training data to use (e.g., 0.1 for 10%)")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--patience', type=int, default=2,
                        help='patience for early stop')
    parser.add_argument('--lang', type=str, default='java')

    # print arguments
    args = parser.parse_args()
    return args


def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


def build_config_tokenizer_model_gpt(args):
    model_classes = {
        "CodeGPT": (GPT2Config, GPT2Tokenizer, GPT2LMHeadModel)
    }

    # Load pre-trained model
    config_class, tokenizer_class, model_class = model_classes[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case,
        sep_token='concode_elem_sep', bos_token='<s>', eos_token='</s>',
        pad_token='<pad>', unk_token='<|UNKNOWN|>',
    )

    model = model_class.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    update_config(model, tokenizer)
    config = model.config

    return config, tokenizer, model


def build_tokenizer_model_mamba(args):
    model_classes = {
        "mamba": (AutoTokenizer, MambaLMHeadModel),
        "mamba2": (AutoTokenizer, MambaLMHeadModel),
    }

    # special_tokens = get_special_tokens(args.lit_file)

    tokenizer_class, model_class = model_classes[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        "../huggingface/EleutherAI/gpt-neox-20b",
        do_lower_case=args.do_lower_case,
        sep_token='concode_elem_sep',
    )

    model = model_class.from_pretrained(args.model_name_or_path)

    return tokenizer, model


def build_tokenizer_model_mamba_hf(args):
    model_classes = {
        "mamba-hf": (AutoTokenizer, MambaForCausalLM),
        "mamba2-hf": (AutoTokenizer, AutoModelForCausalLM)
    }

    tokenizer_class, model_class = model_classes[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        "../huggingface/EleutherAI/gpt-neox-20b",
        do_lower_case=args.do_lower_case,
        sep_token='concode_elem_sep',
    )

    model = model_class.from_pretrained(args.model_name_or_path)
    # model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def build_model_for_classification(args):
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        num_labels=2 
    )

    model.config.pad_token_id = tokenizer.pad_token_id #
    
    model.resize_token_embeddings(len(tokenizer)) #
    config = model.config

    return config, tokenizer, model

def build_mamba_model_for_classification(args):
    print("\n--- Using FINAL CORRECTED version: Manually resizing and loading weights ---\n")

    tokenizer_path = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, do_lower_case=args.do_lower_case
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MambaConfig.from_pretrained(args.model_name_or_path)

    model = MambaForCausalLM(config)

    model.resize_token_embeddings(len(tokenizer))

    weights_path = os.path.join(args.model_name_or_path, "pytorch_model.bin")
    pretrained_state_dict = torch.load(weights_path, map_location="cpu")

    keys_to_remove = ["backbone.embedding.weight", "lm_head.weight"] 
    for key in keys_to_remove:
        if key in pretrained_state_dict:
            del pretrained_state_dict[key]
            print(f"--- Manually removed '{key}' from pretrained weights ---")

    model.load_state_dict(pretrained_state_dict, strict=False)
    print("--- Successfully loaded compatible weights ---")

    model.config.pad_token_id = tokenizer.pad_token_id
    
    return config, tokenizer, model