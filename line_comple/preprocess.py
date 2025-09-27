import logging
import gc
import json
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, args, tokenizer, mode, file_name):
        self.inputs = []
        with open(file_name) as f:
            data = f.readlines()

        if args.sampled_num > 0 and "train" in file_name:
            data = random.sample(data, args.sampled_num)

        length = len(data)
        logger.info("Data size: %d" % (length))
        input_ids = []
        for idx, x in tqdm(enumerate(data), total=length):
            # if idx > 10:
            #     break

            x = x.strip()
            if x.startswith("<s>") and x.endswith("</s>"):
                pass
            else:
                x = "<s> " + x + " </s>"
            try:
                input_ids.extend(tokenizer.encode(x))
            except Exception:
                pass

        del data
        gc.collect()

        length_token = len(input_ids)
        logger.info(f"tokens: {length_token}")

        for i in range(0, length_token - args.block_size, args.block_size):
            self.inputs.append(input_ids[i: i + args.block_size])
        del input_ids
        gc.collect()

        logger.info(f"lines: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class lineEvalDataset(Dataset):
    def __init__(self, args, tokenizer, mode, file_name):
        with open(file_name) as f:
            datas = f.readlines()

        length = len(datas)
        logger.info("Data size: %d" % (length))
        self.inputs = []
        self.gts = []

        max_source_length = args.block_size - args.max_target_length

        for idx, data in tqdm(enumerate(datas), total=length):
            data = json.loads(data.strip())
            self.inputs.append(tokenizer.encode(data["input"])[-max_source_length:])
            self.gts.append(data["gt"])

            # if idx > 10:
            #     break

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]



