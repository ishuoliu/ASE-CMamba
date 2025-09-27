import os
import math
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from fuzzywuzzy import fuzz
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from util import parse_args, set_seed, build_tokenizer_model_mamba_hf
from model_hf import MambaHFModel
from preprocess import load_csn_data

logger = logging.getLogger(__name__)


def train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.batch_size)
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 1
    args.log_steps = args.gradient_accumulation_steps

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num epoch = %d", args.epochs)

    best_loss = 1e6
    patience = 0
    model.train()

    for epoch in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            ids, masks = batch

            inputs, labels = (ids, ids)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            loss = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.log_steps == 0:
                cur_loss = round(tr_loss / tr_num, 5)
                # logger.warning(f"epoch {epoch} step {step + 1} loss {cur_loss}")
                bar.set_description(f"epoch {epoch} step {step + 1} loss {cur_loss}")

                # if (cur_loss < best_loss):
                #     best_loss = cur_loss
                #     patience = 0
                # else:
                #     patience += 1
                #     if patience >= args.patience:
                #         logger.info('patience greater than {}, early stop!'.format(args.patience))
                #         return

                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            # truncate the gradient, used to prevent exploding gradient.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # save model after save_steps.
            if (step + 1) % args.save_steps == 0:


                checkpoint_prefix = f"checkpoint-best-loss-{epoch}"
                output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_file = os.path.join(output_dir, f"model.bin")
                save_content = {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(save_content, output_file)



def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    gpu_list = args.visible_devices.split(',')
    args.gpu_list = [int(x) for x in gpu_list]

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # set logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # set seed
    set_seed(args.seed)

    # build model
    tokenizer, encoder = build_tokenizer_model_mamba_hf(args)

    model = MambaHFModel(encoder, args)
    # print(model)

    print("From scratch")
    for name, param in model.named_parameters():
        if len(param.shape) < 2:
            torch.nn.init.kaiming_normal_(param.unsqueeze(0), a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_normal_(param, a=math.sqrt(5))

    # assert False


    # print("Use pretrain")
    # checkpoint_prefix = 'checkpoint-best-loss-1/model.bin'
    # pretrain_dir = "../pretrained_mamba/mamba2/python/130m/lr5e-5"
    # output_dir = os.path.join(pretrain_dir, checkpoint_prefix)
    # checkpoint = torch.load(output_dir)
    # model.load_state_dict(checkpoint['model_state_dict'])

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_list)

    dataset = load_csn_data(args, tokenizer)
    train(args, dataset, model)


if __name__ == "__main__":
    main()
