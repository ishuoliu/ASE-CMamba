import json
import logging
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 ):
        self.idx = idx
        self.source = source


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())

            nl = ' '.join(js['docstring_tokens']).replace('\n', ' ')
            nl = ' '.join(nl.strip().split())
            # new_source = nl + ' ' + code
            new_source = code

            examples.append(
                Example(idx=idx, source=new_source)
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 source_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask


def convert_examples_to_features(examples, tokenizer, args):
    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples)):

        # if (example_index > 100):
        #     break

        source_tokens = tokenizer.tokenize(example.source)[:args.max_length - 2]
        source_tokens = [tokenizer.bos_token] + source_tokens + [tokenizer.sep_token]

        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        if example_index < 1:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))

            logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
            logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
            logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                source_mask,
            )
        )
    return features


def load_csn_data(args, tokenizer):
    train_file = f"../dataset/codesearchnet/{args.lang}/train.jsonl"
    eval_file = f"../dataset/codesearchnet/{args.lang}/valid.jsonl"
    test_file = f"../dataset/codesearchnet/{args.lang}/test.jsonl"

    train_example = read_examples(train_file)
    eval_example = read_examples(eval_file)
    test_example = read_examples(test_file)

    total_example = []
    total_example.extend(train_example)
    total_example.extend(eval_example)
    total_example.extend(test_example)

    total_features = convert_examples_to_features(total_example, tokenizer, args)

    all_ids = torch.tensor([f.source_ids for f in total_features], dtype=torch.long)
    all_mask = torch.tensor([f.source_mask for f in total_features], dtype=torch.long)

    data = TensorDataset(all_ids, all_mask)
    return data
