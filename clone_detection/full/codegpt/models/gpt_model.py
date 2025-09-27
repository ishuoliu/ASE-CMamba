import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, basemodel, args):
        super(GPTModel, self).__init__()
        self.args = args
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