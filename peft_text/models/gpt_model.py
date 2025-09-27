import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, TaskType, get_peft_model, IA3Config


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][:] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class PEFTGPTModel(nn.Module):
    def __init__(self, basemodel, args):
        super(PEFTGPTModel, self).__init__()
        self.args = args

        if args.peft_name in ["lora"]:
            # peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=32,
            #                          lora_dropout=0.1, modules_to_save=["wte", "lm_head"])
            peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=32,
                                     lora_dropout=0.1)
        elif args.peft_name in ["ia3"]:
            # peft_config = IA3Config(peft_type="IA3",
            #                         task_type="FEATURE_EXTRACTION",
            #                         modules_to_save=["wte", "lm_head"])
            peft_config = IA3Config(peft_type="IA3",
                                    task_type="FEATURE_EXTRACTION")
        else:
            assert False, "Invalid PEFT Name!"

        basemodel = get_peft_model(basemodel, peft_config)
        basemodel.print_trainable_parameters()

        self.model = basemodel

    def forward(self, inputs, attn_mask, loss_mask):
        outputs = self.model(inputs, attention_mask=attn_mask)
        logits = outputs[0]
        labels = inputs

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])

        return loss

    def generate(self, inputs, tokenizer):
        max_gen_len = self.args.max_target_length
        beam_size = self.args.beam_size

        m = torch.nn.LogSoftmax(dim=-1)
        outputs = self.model(inputs)[1]
        p = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        for i in range(inputs.shape[0]):
            # Compatible with transformers version 3.3.0 and 4.13.0
            past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                    outputs]
            past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in past]
            # context_mask=source_mask[i:i+1,:].expand(beam_size,-1)
            beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
            input_ids = None
            for _ in range(max_gen_len):
                if beam.done():
                    break
                input_ids = beam.getCurrentState()
                # context_mask=torch.cat((context_mask,input_ids*0+1),-1)
                # mask=context_mask.unsqueeze(0).unsqueeze(-2).unsqueeze(-2).expand(self.config.n_layer, -1, -1, -1, -1)
                transformer_outputs = self.model(input_ids, past_key_values=past_hidden)
                out = m(transformer_outputs[0][:, -1, :]).data
                # out = self.lsm(self.lm_head(transformer_outputs[0][:,-1,:])).data
                beam.advance(out)
                past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                        transformer_outputs[1]]
                past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:beam_size]

            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (max_gen_len - len(p))).view(1, -1) for p in pred]
            p.append(torch.cat(pred, 0).unsqueeze(0))
        p = torch.cat(p, 0)

        return p




