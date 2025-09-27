import os
import re
import torch
import logging
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from util import parse_args, set_seed, build_config_tokenizer_model_gpt
from models import PEFTGPTModel, GPTModel
from preprocess import TrainDataset, lineEvalDataset


logger = logging.getLogger(__name__)


def train(args, train_dataset, eval_dataset, model):
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

    best_ppl = 1e6
    patience = 0
    model.train()

    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            inputs, labels = (batch, batch)
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
                bar.set_description(f"epoch {epoch} step {step + 1} loss {cur_loss}")
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
                results = evaluate(args, eval_dataset, model)

                if results["eval_ppl"] < best_ppl:
                    patience = 0
                    best_ppl = results["eval_ppl"]
                    checkpoint_prefix = "checkpoint-best-ppl"
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

                else:
                    patience += 1
                    if patience >= args.patience:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0
    batch_num = 0
    model.eval()

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            loss = model(inputs, labels)

        eval_loss += loss.item()
        batch_num += 1

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)

    result = {
        "eval_ppl": eval_ppl
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test_line(args, tokenizer, test_dataset, model):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or
                tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    model.eval()

    if args.lang == "python":
        break_ids = [tokenizer.sep_token_id]
    elif args.lang == "java":
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'),
                     tokenizer.convert_tokens_to_ids('Ġ{')]
    else:
        assert False

    preds = []
    gts = []
    edit_sim = 0.0
    em = 0.0
    for step, (inputs, gt) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            p = model.generate_line_preds(inputs, break_ids)

            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.lang == "python":
                    text = DecodeIds(t).strip("<EOL>").strip()
                elif args.lang == "java":
                    text = DecodeIds(t).strip("{").strip()
                else:
                    assert False

                preds.append(text)
                gts.append(gt[0])

                edit_sim += fuzz.ratio(text, gt[0])
                if text.split() == gt[0].split():
                    em += 1

    total = len(preds)
    logger.info("  %s = %s " % ("Edit sim", str(round(edit_sim / total, 2))))
    logger.info("  %s = %s " % ("EM", str(round(em / total * 100, 2))))


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
    config, tokenizer, encoder = build_config_tokenizer_model_gpt(args)

    pre_model = GPTModel(encoder, args)
    if args.use_pretrain:
        print("Use pretrain")
        checkpoint_prefix = args.checkpoint_prefix
        pretrain_dir = args.pretrain_dir

        print(checkpoint_prefix)
        print(pretrain_dir)

        output_dir = os.path.join(pretrain_dir, checkpoint_prefix)
        checkpoint = torch.load(output_dir)
        pre_model.load_state_dict(checkpoint['model_state_dict'])

    model = PEFTGPTModel(pre_model.model, args)
    logger.info("Training/evaluation parameters %s", args)

    print(model)
    # assert False


    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_list)

    # Training
    if args.do_train:
        train_dataset = TrainDataset(args, tokenizer, "train", args.train_data_file)
        eval_dataset = TrainDataset(args, tokenizer, "eval", args.eval_data_file)
        train(args, train_dataset, eval_dataset, model)

    if args.do_test:
        test_dataset = lineEvalDataset(args, tokenizer, "test", args.test_data_file)

        checkpoint_prefix = 'checkpoint-best-ppl/model.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_line(args, tokenizer, test_dataset, model)


if __name__ == "__main__":
    main()
