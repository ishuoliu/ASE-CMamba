import json
import torch
from torch.utils.data import Dataset
import random
import logging

logger = logging.getLogger(__name__)

def load_code_snippets(jsonl_path):
    code_snippets = {}
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            code_snippets[data["idx"]] = data["func"]
    return code_snippets

class CloneDataset(Dataset):
    def __init__(self, args, tokenizer, file_path, code_snippets_path):
        self.args = args
        self.tokenizer = tokenizer

        logger.info(f"Loading code snippets from {code_snippets_path}")
        code_snippets = load_code_snippets(code_snippets_path)

        self.pairs = []
        self.labels = []
        logger.info(f"Loading dataset from {file_path}")
        with open(file_path, 'r') as f:
            for line in f:
                id1, id2, label = line.strip().split('\t')
                code1 = code_snippets.get(id1)
                code2 = code_snippets.get(id2)
                
                if code1 and code2:
                    self.pairs.append((code1, code2))
                    self.labels.append(int(label))
        
        if hasattr(args, 'train_data_ratio') and args.train_data_ratio < 1.0 and 'train' in file_path:
            num_samples = int(len(self.labels) * args.train_data_ratio)

            combined = list(zip(self.pairs, self.labels))
            random.shuffle(combined) 

            if num_samples > 0:
                sampled_combined = combined[:num_samples]
                self.pairs, self.labels = zip(*sampled_combined)
            else: 
                self.pairs, self.labels = [], []

            self.pairs = list(self.pairs)
            self.labels = list(self.labels)

            logger.info(f"Using {args.train_data_ratio * 100:.2f}% of training data: {len(self.labels)} examples.")

    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        code1, code2 = self.pairs[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            code1, 
            code2, 
            truncation=True, 
            max_length=self.args.block_size, 
            padding='max_length', 
            return_tensors="pt"
        )
        
        # Squeeze tensors to remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item