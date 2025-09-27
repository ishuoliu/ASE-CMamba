import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class CustomModelForCloneClassification(nn.Module):
    def __init__(self, basemodel, args):
        super(CustomModelForCloneClassification, self).__init__()
        self.args = args
        self.model = basemodel 

        hidden_size = self.model.config.hidden_size
        
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs, attn_mask, labels):
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attn_mask,
            output_hidden_states=True 
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        
        pooled_output = last_hidden_state[:, -1, :]
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return loss, logits