import os
import torch
import logging
import json
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from util import parse_args, set_seed, build_config_tokenizer_model_gpt
from preprocess import concodeDataset
from models import GPTModel
from bleu import compute_bleu, _bleu


logger = logging.getLogger(__name__)


def train(args, tokenizer, train_dataset, eval_dataset, test_dataset, model):
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
        for step, (batch, token_labels) in enumerate(bar):
            inputs = batch.to(args.device)
            attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=args.device)
            loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=args.device)
            loss = model(inputs, attn_mask, loss_mask)

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

                checkpoint_prefix = "checkpoint-best-ppl"
                output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                test(args, tokenizer, test_dataset, args.test_data_file, model)

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

    for (batch, token_labels) in tqdm(eval_dataloader, total=len(eval_dataloader)):

        inputs = batch.to(args.device)
        attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=args.device)
        loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=args.device)

        with torch.no_grad():
            loss = model(inputs, attn_mask, loss_mask)

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


def test(args, tokenizer, test_dataset, test_data_file, model):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)

    model.eval()
    preds = []

    for step, (batch, token_labels) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        inputs = batch.to(args.device)
        with torch.no_grad():
            p = model.generate(inputs, tokenizer)


            for pred in p:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                # print(text)
                preds.append(text)

    golds = []
    datas = open(test_data_file).readlines()
    for idx, x in enumerate(datas):
        x = json.loads(x)
        golds.append(x["code"])

        # if idx > 10:
        #     break

    # print(preds)
    # print(golds)
    # assert False

    assert len(preds) == len(golds)

    EM = []
    # for pred, gold in zip(preds, golds):
    #     EM.append(pred.split() == gold.split())

    with open(os.path.join(args.output_dir, "test.output"), 'w') as f, \
            open(os.path.join(args.output_dir, "test.gold"), 'w') as f1:
        for pred, gold in zip(preds, golds):
            f.write(pred + '\n')
            f1.write(gold + '\n')
            EM.append(pred.split() == gold.split())

    bleu_score = round(_bleu(os.path.join(args.output_dir, "test.gold"),
                             os.path.join(args.output_dir, "test.output")), 2)
    # bleu_score = round(100 * bleu_score, 2)
    EM = round(np.mean(EM) * 100, 2)

    logger.info("  %s = %s " % ("BLEU-4", str(bleu_score)))
    logger.info("  %s = %s " % ("EM", str(EM)))









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

    model = GPTModel(encoder, args)
    logger.info("Training/evaluation parameters %s", args)

    # print(model)
    # assert False

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_list)

    # Training
    if args.do_train:
        train_dataset = concodeDataset(args, tokenizer, args.train_data_file, "train")
        eval_dataset = concodeDataset(args, tokenizer, args.eval_data_file, "train")


        test_dataset = concodeDataset(args, tokenizer, args.test_data_file, "test")
        train(args, tokenizer, train_dataset, eval_dataset, test_dataset, model)

        # train(args, train_dataset, eval_dataset, model)


    if args.do_test:
        test_dataset = concodeDataset(args, tokenizer, args.test_data_file, "test")


        checkpoint_prefix = 'checkpoint-best-ppl/model.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        test(args, tokenizer, test_dataset, args.test_data_file, model)



if __name__ == "__main__":
    main()
