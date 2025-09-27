# mamba_main.py

import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from util import parse_args, set_seed, build_mamba_model_for_classification
from preprocess import CloneDataset
from models import MambaModel
import sys

logger = logging.getLogger(__name__)


def compute_metrics(labels, preds):
    """计算F1, Precision, Recall和Accuracy"""
    predictions = np.argmax(preds, axis=1)
    accuracy = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def train(args, tokenizer, train_dataset, eval_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 1
    args.log_steps = args.gradient_accumulation_steps

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
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

            # save model after save_steps.
            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, eval_dataset, model)

                if results["f1"] > best_f1:
                    patience = 0
                    best_f1 = results["f1"]
                    logger.info(f"  New best f1 score: {best_f1:.4f}")

                    checkpoint_prefix = "checkpoint-best-f1"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_file = os.path.join(output_dir, f"model.bin")
                    save_content = {'model_state_dict': model_to_save.state_dict()}
                    torch.save(save_content, output_file)
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
    all_logits = []
    all_labels = []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch['input_ids'].to(args.device)
        attn_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)

        with torch.no_grad():
            _, logits = model(inputs, attn_mask, labels)

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    result = compute_metrics(all_labels, all_logits)

    logger.info("***** Eval results *****")
    for key, value in sorted(result.items()):
        logger.info(f"  {key} = {value:.4f}")

    return result


def test(args, test_dataset, model):
    logger.info("  " + "***** Running test *****")
    # The test logic is identical to evaluation logic
    results = evaluate(args, test_dataset, model)
    logger.info("***** Test results *****")
    for key, value in sorted(results.items()):
        logger.info(f"  {key} = {value:.4f}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # set logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        stream=sys.stdout) 
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # set seed
    set_seed(args.seed)

    # build model
    config, tokenizer, encoder = build_mamba_model_for_classification(args)
    model = MambaModel(encoder, args)
    logger.info("Training/evaluation parameters %s", args)

    if args.use_pretrain:
        # This part for loading a pre-trained checkpoint might need adjustment
        # depending on whether the pre-trained model was for classification
        # or just language modeling. The current setup assumes it's for classification.
        logger.info("Loading pre-trained model from %s", args.pretrain_dir)
        checkpoint_prefix = args.checkpoint_prefix
        pretrain_dir = args.pretrain_dir
        output_dir = os.path.join(pretrain_dir, checkpoint_prefix)
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(args.device)
    if args.n_gpu > 1:
        gpu_list = [int(x) for x in args.visible_devices.split(',')]
        model = torch.nn.DataParallel(model, device_ids=gpu_list)

    # Training
    if args.do_train:
        train_dataset = CloneDataset(args, tokenizer, args.train_data_file, args.code_snippets_path)
        eval_dataset = CloneDataset(args, tokenizer, args.eval_data_file, args.code_snippets_path)
        train(args, tokenizer, train_dataset, eval_dataset, model)

    # Testing
    if args.do_test:
        test_dataset = CloneDataset(args, tokenizer, args.test_data_file, args.code_snippets_path)

        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        checkpoint = torch.load(output_dir)

        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        test(args, test_dataset, model)


if __name__ == "__main__":
    main()