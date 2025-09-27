import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import sys
sys.path.append("../")
from reform_peft import LoraConfig, TaskType, get_peft_model, IA3Config


class MambaModel(nn.Module):
    def __init__(self, basemodel, args, eos_ids):
        super(MambaModel, self).__init__()
        self.args = args
        if args.peft_name == "lora":
            peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=32,
                                     lora_dropout=0.1,
                                     target_modules=["x_proj"])
            basemodel = get_peft_model(basemodel, peft_config)
            basemodel.print_trainable_parameters()

        self.model = basemodel
        self._eos = eos_ids

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

    # def generate_line_preds(self, inputs):
    #     outputs = self.model(input_ids=inputs[:, :-1])[0]
    #     print(outputs.shape)



    def generate_line_preds(self, inputs):
        # print(inputs)
        # print(inputs.shape)

        # print(self._eos)
        # print(self._eos[0])

        fn = lambda eos_id: self.model.generate(
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
            eos_token_id=eos_id,
        )

        out = []
        # print(self._eos)
        eos_len = len(self._eos)

        for i in range(eos_len):
            out.append(fn(eos_id=self._eos[i]))

        # print(out.sequences[0])
        # print(out.sequences[0][inputs.shape[1]: ])
        # print(out.sequences[0][inputs.shape[1]: ].shape)
        # assert False

        preds = out[0].sequences[0][inputs.shape[1]: ]
        for i in range(1, eos_len):
            senten = out[i].sequences[0][inputs.shape[1]: ].shape[0]
            if senten < preds.shape[0]:
                preds = out[i].sequences[0][inputs.shape[1]: ]

        return preds

        # print(preds)
        #
        #
        # assert False

        # sentence = []
        # for pred in preds:
        #     sentence.append(pred)
        #     if pred in self._eos:
        #         break
        #
        # zero = torch.cuda.LongTensor(1).fill_(0)
        # sen = [torch.cat([x.view(-1)] for x in sentence)]
        #
        # # sen = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
        # #         sentence]
        #
        # print(sen)
        # assert False
        #
        # # return
