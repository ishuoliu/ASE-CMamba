# in gpt_model.py

from peft import LoraConfig, TaskType, get_peft_model, IA3Config
import torch
import torch.nn as nn

class PEFTGPTModel(nn.Module):
    def __init__(self, basemodel, args):
        super(PEFTGPTModel, self).__init__()
        self.args = args

        if args.peft_name in ["lora"]:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, 
                r=8, 
                lora_alpha=16,
                lora_dropout=0.1
            )
        elif args.peft_name in ["ia3"]:
            peft_config = IA3Config(
                task_type=TaskType.SEQ_CLS,
            )
        else:
            assert False, "Invalid PEFT Name!"

        basemodel = get_peft_model(basemodel, peft_config)
        basemodel.print_trainable_parameters()
        self.model = basemodel

    def forward(self, inputs, attn_mask, labels):
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attn_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits
