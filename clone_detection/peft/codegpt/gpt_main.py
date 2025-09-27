import os
import torch
import logging
import json
import numpy as np
from tqdm import tqdm

import sys 

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout) 

import transformers
transformers.logging.set_verbosity_error()

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn as nn
from util import parse_args, set_seed, build_config_tokenizer_model_gpt
from preprocess import CloneDataset
from models import PEFTGPTModel


logger = logging.getLogger(__name__)


def compute_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }


def train(args, tokenizer, train_dataset, eval_dataset, test_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 1
    args.log_steps = args.gradient_accumulation_steps

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps * args.max_steps),
                                                num_training_steps=args.max_steps)

    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num epoch = %d", args.num_train_epochs)

    best_f1 = 0
    patience = 0
    model.train()

    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            inputs = batch['input_ids'].to(args.device)
            attn_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            loss, _ = model(inputs, attn_mask, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.log_steps == 0:
                cur_loss = round(tr_loss / tr_num, 5)
                bar.set_description(f"epoch {epoch} step {step + 1} loss {cur_loss}")
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, eval_dataset, model)

                if results["eval_f1"] > best_f1:
                    patience = 0
                    best_f1 = results["eval_f1"]
                    logger.info(f"  New best f1: {best_f1}")
                    checkpoint_prefix = "checkpoint-best-f1"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # For PEFT, use save_pretrained to save the adapter
                    model_to_save.model.save_pretrained(output_dir)
                    logger.info(f"Saving model checkpoint to {output_dir}")

                else:
                    patience += 1
                    if patience >= args.patience:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("  " + "***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch['input_ids'].to(args.device)
        attn_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)

        with torch.no_grad():
            _, logits = model(inputs, attn_mask, labels)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    result = {
        "eval_acc": metrics["acc"],
        "eval_f1": metrics["f1"],
        "eval_precision": metrics["precision"],
        "eval_recall": metrics["recall"],
    }
    logger.info("***** Eval results *****")
    for key, value in result.items():
        logger.info(f"  {key} = {value:.4f}")

    return result


def test(args, tokenizer, test_dataset, model):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    logger.info("  " + "***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        inputs = batch['input_ids'].to(args.device)
        attn_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        with torch.no_grad():
            _, logits = model(inputs, attn_mask, labels)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    result = compute_metrics(all_labels, all_preds)

    logger.info("***** Test results *****")
    for key, value in result.items():
        logger.info(f"  {key} = {value:.4f}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    gpu_list = args.visible_devices.split(',')
    args.gpu_list = [int(x) for x in gpu_list]

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # set seed
    set_seed(args.seed)

    # build model
    config, tokenizer, encoder = build_config_tokenizer_model_gpt(args)
    model = PEFTGPTModel(encoder, args)

    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_list)

    # Training
    if args.do_train:
        train_dataset = CloneDataset(args, tokenizer, args.train_data_file, args.code_snippets_path)
        eval_dataset = CloneDataset(args, tokenizer, args.eval_data_file, args.code_snippets_path)
        test_dataset = CloneDataset(args, tokenizer, args.test_data_file, args.code_snippets_path)
        train(args, tokenizer, train_dataset, eval_dataset, test_dataset, model)

    if args.do_test:
        logger.info("  " + "***** Running final test on the best checkpoint *****")
        test_dataset = CloneDataset(args, tokenizer, args.test_data_file, args.code_snippets_path)

        checkpoint_prefix = 'checkpoint-best-f1'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        
        # Reload the base model and apply the saved adapter
        _, _, base_encoder = build_config_tokenizer_model_gpt(args)
        model = PEFTGPTModel(base_encoder, args)
        model.model.load_adapter(output_dir, adapter_name="default")
        model.to(args.device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_list)

        test(args, tokenizer, test_dataset, model)


if __name__ == "__main__":
    main()