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
from datetime import datetime
import re
import wandb

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, PretrainedConfig
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
from dense_retriever import Retriever
from reasoning_writer import ReasoningWriter

choices = ["A", "B", "C", "D"]

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def setup_model(args):
    device = args.device
    tokenizer_name = "google/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    checkpoint_dir = get_last_checkpoint(args.checkpoint_path)
    config = PretrainedConfig.from_pretrained(os.path.join(checkpoint_dir, "config.json"))
    model = load_state_dict_from_zero_checkpoint(T5ForConditionalGeneration(config), checkpoint_dir)
    model.to(device)
    return model, tokenizer

def setup_searcher(args):
    cache_dir = f"./retriever_cache/{args.dataset}"
    cache_path = os.path.join(cache_dir, f"{args.fold}_{args.knowledge_base}_cache.jsonl")

    ### Setup for Retrieval
    if args.ground_truth_doc:
        print("Load Searcher...")
        cache_path = f"./retriever_cache/{args.dataset}/{args.fold}_{args.knowledge_base}_GT_cache.jsonl"
        if args.knowledge_base == "wikipedia":
            searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
        elif args.knowledge_base == "pubmed":
            searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-bioasq-flat')
        print(f"Cache Found {cache_path}")
        with open(cache_path, 'r') as f:
            cache_dumps = f.readlines()
        cache_dumps = [json.loads(_cache.strip()) for _cache in cache_dumps]
        cache_db = dict()
        for cache_dump in cache_dumps:
            cache_db[cache_dump["id"]] = cache_dump["documents"]
    else:
        if args.knowledge_base is not None:
            print("Load Searcher...")
            if args.knowledge_base == "wikipedia":
                searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
            elif args.knowledge_base == "pubmed":
                searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-bioasq-flat')

            # Load cache
            print(f"Cache exist? {cache_path}")
            if os.path.exists(cache_path):
                print(f"Cache Found {cache_path}")
                with open(cache_path, 'r') as f:
                    cache_dumps = f.readlines()
                cache_dumps = [json.loads(_cache.strip()) for _cache in cache_dumps]
                cache_db = dict()
                for cache_dump in cache_dumps:
                    cache_db[cache_dump["id"]] = cache_dump["documents"]
            else:
                cache_db = None
        else:
            if args.knowledge_base is not None: raise NotImplementedError
            searcher = None
            cache_db = None
    return searcher, cache_db

def setup_dataset(args):
    ### Setup for Dataset
    dataset_path = f"./data/{args.dataset}"
    filename = f"{args.fold}.json"
    with open(os.path.join(dataset_path, filename)) as f:
        dataset = [json.loads(data) for data in f.readlines()]
    return dataset

def setup_output_dir (args):
    ### Setup for output directory

    now = datetime.now()
    formatted_date = now.strftime("%Y%m%d")

    if args.knowledge_base is not None:
        output_dir = f"whitebox_lm_output/{args.dataset}/{args.reasoner_base}/{formatted_date}/{args.fold}_{args.knowledge_base}_{args.retriever_type}_ndoc{args.n_docs}"
        if args.retriever_type == "dense":
            output_dir += f"_path={os.path.basename(args.dense_retriever_path)}"
            args.dense_retriever_path = os.path.join(args.dense_retriever_path, "model")
        if args.n_cands > 0:
            output_dir += f"_ncands{args.n_cands}"
        if args.ground_truth_doc:
            output_dir += "_GT"
    else:
        output_dir = f"whitebox_lm_output/{args.dataset}/{args.reasoner_base}/{formatted_date}/{args.fold}"
    if args.greedy:
        output_dir += "_greedy"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Save to {output_dir}")
    ########
    return output_dir

def setup_prompt(args):

    if args.dataset == "medqa_usmle_hf":
        domain = "medical"
    elif args.dataset == "strategyqa":
        domain = "common"
    elif args.dataset == "obqa":
        domain = "commonsense"
    else:
        raise NotImplementedError

    if args.knowledge_base is not None:
        prompt_template_part1 = \
            f"The following are multiple-choice questions about {domain} knowledge. Generate a step-by-step explanations for each question with given {domain} knowledge.\nQuestion: {{input}}\nKnowledge: {{knowledge}}"
        prompt_template_part2 = "\nExplanation:"
        if args.dataset in ["strategyqa"]:
            data_template = "{} A. {} B. {}"
        else:
            data_template = "{} A. {} B. {} C. {} D. {}"
    else:
        prompt_template_part2 = ""
        prompt_template_part1 = f"The following are multiple-choice questions about {domain} knowledge. Generate a step-by-step explanations for each question:\nQuestion: {{input}}\nExplanation:"
        
        if args.dataset in ["strategyqa"]:
            data_template = "{} A. {} B. {}"
        else:
            data_template = "{} A. {} B. {} C. {} D. {}"
    return prompt_template_part1, prompt_template_part2, data_template


def process_knowledge(args, knowledges, from_cache=False):
    documents = []
    if args.knowledge_base == "wikipedia":
        for knowledge in knowledges:
            if from_cache:
                documents.append(knowledge.raw())
            else:
                documents.append(knowledge.raw)
    elif args.knowledge_base == "pubmed":
        for knowledge in knowledges:
            doc = json.loads(knowledge.raw())
            documents.append(doc["text"])
    return documents

def load_pseudo_reasoning(args, data_idx):
    with open(os.path.join(args.pseudo_reasoning_path, str(data_idx).zfill(5) + ".json"), 'r') as f:
        response = json.load(f)
    q = response["choices"][0]["message"]["content"]
    return q

def load_keywords(args, data_idx):
    with open(os.path.join(args.keywords_path, str(data_idx).zfill(5) + ".json"), 'r') as f:
        response = json.load(f)
    q = response["choices"][0]["message"]["content"]
    return q

def main(args):
    reasoner_base = args.checkpoint_path.split("/")[2]
    print(f"Reasoner base: {reasoner_base}")
    args.reasoner_base = reasoner_base

    device = "cuda:0"
    args.device = device
    model, tokenizer = setup_model(args) # Model Setup
    searcher, cache_db = setup_searcher(args) # Sparse Search Setup
    dataset = setup_dataset(args)
    output_dir = setup_output_dir(args)
    dense_retriever = Retriever(args)
    prompt_template_part1, prompt_template_part2, data_template = setup_prompt(args)

    writer = ReasoningWriter(args)

    current_id = 0
    max_new_tokens = 1024
    if args.max_doc > 0:
        MAX_DOC = args.max_doc
    else:
        MAX_DOC = 100

    accs = []
    for data_idx, data in enumerate(tqdm(dataset)):
        out_filename = str(current_id).zfill(5) + ".json"

        if args.lm_type == "gpt":
            max_inpt_tokens = tokenizer.model_max_length
            max_inpt_tokens -= max_new_tokens
        else:
            max_inpt_tokens = 2048
            prompt2_tokens = tokenizer(prompt_template_part2, return_tensors="pt", add_special_tokens=True).to(device)
            max_inpt_tokens = max_inpt_tokens - prompt2_tokens.input_ids.shape[-1]

        if args.knowledge_base is None:
            if args.dataset in ["strategyqa"]:
                q = data_template.format(data["sent1"], data["ending0"], data["ending1"])
            else:
                q = data_template.format(data["sent1"], data["ending0"], data["ending1"], data["ending2"], data["ending3"])
            prompt = prompt_template_part1.format(input=q)
            inpts = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            inpts_list = [inpts]
            knowledge = ''
        else:
            if args.dataset in ["strategyqa"]:
                q = data_template.format(data["sent1"], data["ending0"], data["ending1"])
            else:
                q = data_template.format(data["sent1"], data["ending0"], data["ending1"], data["ending2"], data["ending3"])
            search_q = q
            if cache_db is not None:
                documents = [searcher.doc(hit_id) for hit_id in cache_db[data_idx]]
            else:
                documents = [hit for hit in searcher.search(search_q, k=MAX_DOC)]

            documents = process_knowledge(args, documents, from_cache=cache_db is not None)
            if args.max_doc > 0:
                documents = documents[:args.max_doc]

            if args.ground_truth_doc:
                knowledge = documents[:args.n_docs]
            else:
                if args.retriever_type == "dense":
                    documents = dense_retriever.retrieve(search_q, documents, n=args.n_docs, n_cands=args.n_cands)
                    knowledge = documents[:args.n_docs]
                elif args.retriever_type == "sparse":
                    knowledge = documents[:args.n_docs]
            knowledges = [knowledge]

            prompt = prompt_template_part1.format(input=q, knowledge=' '.join(knowledge) if type(knowledge) == list else knowledge)
            prompt2 = prompt_template_part2
            inpts = tokenizer(prompt, max_length=max_inpt_tokens, truncation=True, return_tensors="pt", add_special_tokens=False).to(device)
            inpts.input_ids = torch.cat([inpts.input_ids, prompt2_tokens.input_ids], dim=-1)
            inpts.attention_mask = torch.cat([inpts.attention_mask, prompt2_tokens.attention_mask], dim=-1)

        print(tokenizer.decode(inpts.input_ids[0].tolist()))

        # prompt = '''
        #     The following are multiple-choice questions about medical knowledge. Generate a step-by-step explanations for each question:
        #     Question: A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?
        #     (A) Refuse to dictate the operative report (B) Tell the attending that he cannot fail to disclose this mistake (C) Report the physician to the ethics committee (D) Disclose the error to the patient and put it in the operative report
        #     Explanation:
        # '''
        
        votes = []
        gold_label = choices[data["label"]]
        rationales = []
        pred_choices = []

        if args.lm_type == "t5":
            decoder_input_ids = tokenizer("", return_tensors="pt").input_ids
            decoder_input_ids = model._shift_right(decoder_input_ids).to(device)
            if args.greedy:
                gen = model.generate(input_ids=inpts.input_ids[:, -max_inpt_tokens:],
                                        decoder_input_ids=decoder_input_ids,
                                        attention_mask=inpts.attention_mask[:, -max_inpt_tokens:], 
                                        pad_token_id=tokenizer.eos_token_id, 
                                        max_new_tokens=max_new_tokens, 
                                        do_sample=False)
            else:
                gen = model.generate(input_ids=inpts.input_ids[:, -max_inpt_tokens:],
                                        decoder_input_ids=decoder_input_ids,
                                        attention_mask=inpts.attention_mask[:, -max_inpt_tokens:], 
                                        pad_token_id=tokenizer.eos_token_id, 
                                        max_new_tokens=max_new_tokens, 
                                        top_k=50, top_p=0.95, do_sample=True, 
                                        num_return_sequences=args.self_consistency)
        actual_prompt = tokenizer.decode(inpts.input_ids[0])
        for _gen in gen:
            pred = tokenizer.decode(_gen, skip_special_tokens=True)
            choice = pred[-1]
            if choice in choices:
                votes.append(choice)

            rationales.append(pred)
            pred_choices.append(choice)

        print(pred)

        if len(votes) == 0:
            pred = "null"
        else:
            pred = most_common(votes)

        if gold_label == pred:
            accs.append(1)
        else:
            accs.append(0)
        print(votes)

        cprint(f"Prediction: {pred}", 'blue', end=' ')
        cprint(f"Answer: {gold_label}", 'magenta')

        writer.write(args, data_idx, rationales, pred_choices, gold_label, knowledge)
        print(sum(accs) / len(accs))
        print()
        current_id += 1
    acc = sum(accs) / len(accs)
    print(acc)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_base', type=str, default="wikipedia", choices=["None", "wikipedia", "pubmed"])
    parser.add_argument('--retriever_type', type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument('--retriever_model', type=str, default="colbert", choices=["colbert"])
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="medqa_usmle_hf", choices=["medqa_usmle_hf", "strategyqa","obqa"])
    parser.add_argument('--lm_type', type=str, default="t5", choices=["t5", "gpt"])
    parser.add_argument('--n_docs', type=int, default=1, help="The number of documents to be prompted")
    parser.add_argument('--n_cands', type=int, default=-1, help="The number of documents for reranking")
    parser.add_argument('--ground_truth_doc', action='store_true')
    parser.add_argument('--from_dense', action='store_true', help="Load cached documents from the dense retriever (dpr) instead of BM25")
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--fold', type=str, choices=["train", "valid", "test"], default="test")
    parser.add_argument('--dense_retriever_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--self_consistency', type=int, default=5)
    parser.add_argument('--max_doc', type=int, default=-1)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.knowledge_base == "None": args.knowledge_base = None
    
    main(args)