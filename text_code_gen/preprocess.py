import logging
import gc
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class concodeDataset(Dataset):
    def __init__(self, args, tokenizer, file_name, mode):
        self.args = args
        self.mode = mode
        self.block_size = args.block_size

        self.inputs = []
        self.token_labels = []
        with open(file_name) as f:
            data = f.readlines()

        length = len(data)
        logger.info("Data size: %d" % (length))
        for idx, x in tqdm(enumerate(data), total=length):
            x = json.loads(x)
            code = tokenizer.encode(x["code"])
            nl = tokenizer.encode(x["nl"])

            input_ids, input_labels = self.pad_and_get_mask(code, nl, tokenizer)
            self.inputs.append(input_ids)
            self.token_labels.append(input_labels)

            # print(len(input_ids), len(input_labels))


            # if idx > 10:
            #     break


    def pad_and_get_mask(self, code, nl, tokenizer):
        if self.mode == 'test':
            code = []
        while (len(code) + len(nl) + 2 > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]
        if self.mode == 'train':
            inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
            labels = [1] * len(nl) + [2] * (len(code) + 1) + [0]
        else:
            inputs = nl + [tokenizer.bos_token_id]
            labels = [1] * len(nl) + [2]
            return inputs, labels
        assert len(inputs) <= self.block_size

        # if not self.args.is_mamba:
        pad_len = self.block_size - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len

        assert len(inputs) == len(labels)
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])


