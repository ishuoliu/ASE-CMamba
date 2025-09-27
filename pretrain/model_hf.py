import torch
import torch.nn as nn


class MambaHFModel(nn.Module):
    def __init__(self, basemodel, args):
        super(MambaHFModel, self).__init__()
        self.args = args
        self.model = basemodel

    def forward(self, source_id, target_id):
        outputs = self.model(input_ids=source_id, labels=target_id)
        loss = outputs[0]

        return loss


