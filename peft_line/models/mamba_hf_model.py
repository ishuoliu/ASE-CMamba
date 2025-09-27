import torch
import torch.nn as nn
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
                if self.nextYs[-1][i] in self._eos:
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
            if self.nextYs[-1][i] in self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] in self._eos:
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
                if self.nextYs[-1][i] not in self._eos:
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
                tokens.append(tok)
                if tok in self._eos:
                    break
            sentence.append(tokens)
        return sentence


class PEFTMambaHFModel(nn.Module):
    def __init__(self, basemodel, args, eos_ids):
        super(PEFTMambaHFModel, self).__init__()
        self.args = args
        self._eos = eos_ids

        if args.peft_name == "lora":
            if args.model_type == "mamba-hf":
                peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=32,
                                         lora_dropout=0.1,
                                         target_modules=["in_proj", "x_proj", "out_proj"])
                                         # target_modules=["in_proj", "x_proj", "embeddings", "out_proj"])
                # peft_config = LoraConfig(
                #     r=8,
                #     target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
                #     task_type="CAUSAL_LM",
                #     bias="none"
                # )
            elif args.model_type == "mamba2-hf":
                peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=32,
                                         lora_dropout=0.1,
                                         target_modules=["in_proj", "out_proj"])
                                         # target_modules=["in_proj", "embeddings", "out_proj"])
                # peft_config = LoraConfig(
                #     r=8,
                #     target_modules=["embeddings", "in_proj", "out_proj"],
                #     task_type="CAUSAL_LM",
                #     bias="none"
                # )
            else:
                assert False
        elif args.peft_name == "ia3":
            if args.model_type == "mamba-hf":
                peft_config = IA3Config(peft_type="IA3",
                                        task_type="FEATURE_EXTRACTION",
                                        target_modules=["x_proj", "in_proj", "out_proj"],
                                        feedforward_modules=["out_proj"])
            elif args.model_type == "mamba2-hf":
                peft_config = IA3Config(peft_type="IA3",
                                        task_type="FEATURE_EXTRACTION",
                                        target_modules=["in_proj", "out_proj"],
                                        feedforward_modules=["out_proj"])
            else:
                assert False

        basemodel = get_peft_model(basemodel, peft_config)
        basemodel.print_trainable_parameters()


        self.model = basemodel


    def forward(self, source_id, target_id):
        outputs = self.model(input_ids=source_id, labels=target_id)
        loss = outputs[0]

        return loss

    def generate_line_preds(self, inputs):


        fn = lambda eos_id: self.model.generate(
            input_ids=inputs,
            max_length=inputs.shape[1] + self.args.max_target_length,
            # cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            # enable_timing=False,
            temperature=1.0,
            # top_k=1,
            top_p=1.0,
            # min_p=0.0,
            # do_sample=True,
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



class MambaHFModel(nn.Module):
    def __init__(self, basemodel, args, eos_ids):
        super(MambaHFModel, self).__init__()
        self.args = args
        self._eos = eos_ids
        self.model = basemodel
