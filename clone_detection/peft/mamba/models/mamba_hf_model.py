import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, get_peft_model, IA3Config

class PEFTMambaClassificationModel(nn.Module):
    def __init__(self, basemodel, args):
        super(PEFTMambaClassificationModel, self).__init__()
        self.args = args

        if "mamba2" in args.model_name_or_path.lower():
            print("--- Detected Mamba-2 model. Applying PEFT config for Mamba-2. ---")
            target_modules = ["in_proj", "out_proj"]
        else:
            print("--- Detected original Mamba model. Applying PEFT config for Mamba. ---")
            target_modules = ["in_proj", "x_proj", "out_proj"]
        
        if args.peft_name == "lora":
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                target_modules=target_modules
            )
        elif args.peft_name == "ia3":
            peft_config = IA3Config(
                peft_type="IA3",
                target_modules=target_modules,
                feedforward_modules=target_modules
            )
        else:
            raise ValueError("Unsupported PEFT method. Choose 'lora' or 'ia3'.")
        basemodel = get_peft_model(basemodel, peft_config)
        basemodel.print_trainable_parameters()
        self.model = basemodel
        
        hidden_size = self.model.config.hidden_size
        num_labels = 2 
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs, attn_mask, labels=None):
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attn_mask,
            output_hidden_states=True
        )

        last_hidden_states = outputs.hidden_states[-1]

        sequence_lengths = torch.sum(attn_mask, dim=1) - 1
        batch_size = inputs.shape[0]
        last_token_hidden_states = last_hidden_states[torch.arange(batch_size, device=inputs.device), sequence_lengths]

        logits = self.classifier(last_token_hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return loss, logits