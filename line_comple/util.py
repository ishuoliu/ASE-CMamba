import argparse
import random
import numpy as np
import torch
import os
import json
import sys
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, MambaForCausalLM, AutoModelForCausalLM)
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

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

    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")

    parser.add_argument("--model_type", default="CodeGPT", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--visible_devices", default="0", type=str,
                        help="The indices of visible GPUs")

    # parser.add_argument("--max_source_length", default=512, type=int,
    #                     help="The maximum total source sequence length after tokenization. Sequences longer "
    #                          "than this will be truncated, sequences shorter will be padded.")
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
    parser.add_argument("--use_pretrain", action='store_true')
    parser.add_argument("--checkpoint_prefix", default=None, type=str)
    parser.add_argument("--pretrain_dir", default=None, type=str)

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", type=int, default=0.1)

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--patience', type=int, default=2,
                        help='patience for early stop')
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--sampled_num', type=int, default=-1)

    # print arguments
    args = parser.parse_args()
    return args


def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens



def build_config_tokenizer_model_gpt(args):
    model_classes = {
        "CodeGPT": (GPT2Config, GPT2Tokenizer, GPT2LMHeadModel)
    }

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)

    # Load pre-trained model
    config_class, tokenizer_class, model_class = model_classes[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path)

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
        # additional_special_tokens=special_tokens
    )

    print(len(tokenizer))

    model = model_class.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    # print(model)
    # assert False

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
        # sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
        # additional_special_tokens=special_tokens
    )

    print(len(tokenizer))
    # assert False

    model = model_class.from_pretrained(args.model_name_or_path)

    return tokenizer, model


def build_tokenizer_model_mamba_hf(args):
    model_classes = {
        "mamba-hf": (AutoTokenizer, MambaForCausalLM),
        "mamba2-hf": (AutoTokenizer, AutoModelForCausalLM)
    }

    # get special tokens
    # special_tokens = get_special_tokens(args.lit_file)

    tokenizer_class, model_class = model_classes[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        "../huggingface/EleutherAI/gpt-neox-20b",
        # sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
        # additional_special_tokens=special_tokens
    )

    model = model_class.from_pretrained(args.model_name_or_path)
    # model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model
