import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class MambaModel(nn.Module):
    def __init__(self, basemodel, args):
        super(MambaModel, self).__init__()
        self.args = args
        self.model = basemodel

    def forward(self, input_ids, labels):
        lm_logits = self.model(input_ids=input_ids)[0]

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss





