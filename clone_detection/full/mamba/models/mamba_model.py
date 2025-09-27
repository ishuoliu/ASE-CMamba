# mamba_model.py
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class MambaModel(nn.Module):
    def __init__(self, basemodel, args):
        super(MambaModel, self).__init__()
        self.args = args
        self.model = basemodel 
        
        hidden_size = self.model.config.hidden_size
        num_labels = 2

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs, attn_mask, labels):
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attn_mask,
            output_hidden_states=True
        )

        last_hidden_states = outputs.hidden_states[-1] # Shape: (batch_size, sequence_length, hidden_size)

        sequence_lengths = torch.sum(attn_mask, dim=1) - 1 
        batch_size = inputs.shape[0]
        
        last_token_hidden_states = last_hidden_states[torch.arange(batch_size, device=inputs.device), sequence_lengths]

        logits = self.classifier(last_token_hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return loss, logits