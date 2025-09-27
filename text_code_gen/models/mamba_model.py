import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class MambaModel(nn.Module):
    def __init__(self, basemodel, args):
        super(MambaModel, self).__init__()
        self.args = args
        self.model = basemodel

    def forward(self, input_ids, attn_mask, loss_mask):
        logits = self.model(input_ids=input_ids)[0]
        # print(attn_mask.unsqueeze(1).shape)
        # print(logits.shape)

        labels = input_ids

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])

        return loss


    def generate(self, inputs, tokenizer):
        fn = lambda: self.model.generate(
            input_ids=inputs,
            max_length=inputs.shape[1] + self.args.max_target_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

        out = fn()
        preds = out.sequences[0][inputs.shape[1]:]

        return preds

        # print(preds)
        # print(preds.shape)
        # assert False

        # out = []
        # # print(self._eos)
        # eos_len = len(self._eos)
        #
        # for i in range(eos_len):
        #     out.append(fn(eos_id=self._eos[i]))
        #
        # preds = out[0].sequences[0][inputs.shape[1]: ]
        # for i in range(1, eos_len):
        #     senten = out[i].sequences[0][inputs.shape[1]: ].shape[0]
        #     if senten < preds.shape[0]:
        #         preds = out[i].sequences[0][inputs.shape[1]: ]

        # return preds


