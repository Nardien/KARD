import random
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
from termcolor import cprint
import string
from collections import Counter

seed = 42

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration

transformers.utils.move_cache()

class Verifier:
    def __init__(self, args, num_choices=4):
        self.id_to_label = ["A", "B", "C", "D"]
        
        self.completion_template = "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:"
        self.fact_keys = ["cands0", "cands1", "cands2", "cands3"]
        self.device = "cuda"

        gpt = "google/flan-t5-large"
        self.max_new_tokens = 1

        self.tokenizer = AutoTokenizer.from_pretrained(gpt)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model_type = "t5"
        self.model = T5ForConditionalGeneration.from_pretrained(gpt).eval().to(self.device)

        template_path = f"prompts/{args.dataset}/template_cot_5shot.txt"
        instruction_path = f"prompts/{args.dataset}/instruction_cot.txt"

        with open(template_path) as f:
            few_shot_examples = f.readlines()
            self.few_shot_examples = ['\n'.join(example.strip().split('\\n')) + '\n\n' for example in few_shot_examples]

        # print(instruction)
        for ex in self.few_shot_examples:
            print(ex)

        with open(instruction_path) as f:
            instruction = f.readline()
            self.instruction = instruction.strip() + '\n\n'

    def call_model(self, prompt, q, model, tokenizer, device, max_new_tokens=15, model_max_length=None, model_type='gpt'):
        max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length

        inst_inpts = tokenizer(prompt["instruction"], return_tensors="pt", add_special_tokens=False).to(device)
        q_inpts = tokenizer(q, return_tensors="pt").to(device)

        inst_len = inst_inpts.input_ids.shape[-1]
        q_len = q_inpts.input_ids.shape[-1]

        budget = max_inpt_tokens - inst_len - q_len

        input_ids = inst_inpts.input_ids
        attention_mask = inst_inpts.attention_mask

        for example in prompt["few_shot_examples"]:
            ex_inpts = tokenizer(example, return_tensors="pt", add_special_tokens=False).to(device)
            if ex_inpts.input_ids.shape[-1] <= budget:
                input_ids = torch.cat([input_ids, ex_inpts.input_ids], dim=-1)
                attention_mask = torch.cat([attention_mask, ex_inpts.attention_mask], dim=-1)
                budget -= ex_inpts.input_ids.shape[-1]

        input_ids = torch.cat([input_ids, q_inpts.input_ids], dim=-1)
        attention_mask = torch.cat([attention_mask, q_inpts.attention_mask], dim=-1)

        decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(device)
        decoder_input_ids = model._shift_right(decoder_input_ids)
        gen = model.generate(input_ids=input_ids[:, -max_inpt_tokens:],
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask[:, -max_inpt_tokens:], 
                            pad_token_id=tokenizer.eos_token_id, 
                            max_new_tokens=max_new_tokens, 
                            num_beams=1, do_sample=False, return_dict_in_generate=True, output_scores=True,)

        scores = torch.cat(gen.scores, dim=0)
        probs = torch.log_softmax(scores, dim=-1)

        pred = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)

        if pred.startswith("\n\n"):
            pred = pred[2:]

        text = tokenizer.decode(input_ids[0], skip_special_tokens=True) + pred
        return pred, text, probs

    def load_reasoning_path(self, q, cot_data):
        qs = []
        cots = []
        for item in cot_data["choices"]:
            content = item["message"]["content"]
            _q = q + content + " So, the answer is "
            qs.append(_q)
            cots.append(content)
        return qs, cots

    def get_mc_answer(self, tokenizer, probs, pred, answers):
        lprobs = []
        if probs[-1].argmax().item() == tokenizer.eos_token_id:
            last_is_eos = True
        else:
            last_is_eos = False

        if len(pred) != 0:
            if pred[-1] in string.punctuation:
                prob = probs[-1]
            else:
                prob = probs[-2 if last_is_eos else -1]
        else:
            prob = probs[-1]

        for ans in answers:
            lprobs.append(prob[ans])
        pred_idx = np.argmax([lprob.item() for lprob in lprobs])
        return pred_idx

    def __call__(self, data, cot_data):
        prompt = {
            "instruction": self.instruction,
            "few_shot_examples": self.few_shot_examples,
        }

        generate = lambda q, max_new_tokens: self.call_model(prompt, q, model=self.model, tokenizer=self.tokenizer, device=self.device, max_new_tokens=self.max_new_tokens, model_type=self.model_type)
        answers = [self.tokenizer.encode(c)[0] for c in self.id_to_label]

        q = self.completion_template.format(data["sent1"], data["ending0"], data["ending1"], data["ending2"], data["ending3"])
        qs, cots = self.load_reasoning_path(q, cot_data)

        correct_cot = []
        wrong_cot = []
        for q, cot in zip(qs, cots):
            print(cot)
            pred, response, probs = generate(q, max_new_tokens=self.max_new_tokens)
            pred_idx = self.get_mc_answer(self.tokenizer, probs, pred, answers)

            if data["label"] == pred_idx:
                correct_cot.append(cot)
                cprint("Correct", 'green')
            else:
                wrong_cot.append(cot)
                cprint("Wrong", 'red')
            print()
            
        return correct_cot, wrong_cot
