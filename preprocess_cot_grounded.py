import numpy as np
import os
import json
import argparse

from datasets import load_dataset, concatenate_datasets, Dataset
import transformers
from transformers import AutoTokenizer
from verify_cot import Verifier
from tqdm import tqdm
from termcolor import cprint
transformers.utils.move_cache()
from pyserini.search.lucene import LuceneSearcher

import hashlib
def get_hash_value(in_str, in_digest_bytes_size=64, in_return_type='hexdigest'):
    assert 1 <= in_digest_bytes_size and in_digest_bytes_size <= 64
    blake  = hashlib.blake2b(in_str.encode('utf-8'), digest_size=in_digest_bytes_size)
    if in_return_type == 'hexdigest': return blake.hexdigest()
    elif in_return_type == 'number': return int(blake.hexdigest(), base=16)
    return blake.digest()

parser = argparse.ArgumentParser()
parser.add_argument('--knowledge_base', type=str, default="wikipedia", choices=["wikipedia", "pubmed"])
parser.add_argument('--dataset', type=str, default="medqa_usmle_hf", choices=["medqa_usmle_hf", "strategyqa", "obqa"])
parser.add_argument('--with_answer', action='store_true', help="Append answer at the last of dataset")
parser.add_argument('--skip_search', action='store_true', help="skip search but preprocess again")
parser.add_argument('--n_knowledge', type=int, default=1)
args = parser.parse_args()

### Experiment config
model_id = "google/flan-t5-large" # Hugging Face Model Id
text_column = "question" # column of input text is
summary_column = "reasoning" # column of the output text
knowledge_column = "knowledge"
if args.dataset == "medqa_usmle_hf":
    prompt_template_part1 = \
    f"The following are multiple-choice questions about medical knowledge. Generate a step-by-step explanations for each question with given medical knowledge.\nQuestion: {{input}}\nKnowledge: {{knowledge}}"
    data_template = "{} A. {} B. {} C. {} D. {}"
    data_template = "{} A. {} B. {} C. {} D. {}"
elif args.dataset == "strategyqa":
    prompt_template_part1 = \
    f"The following are multiple-choice questions. Generate a step-by-step explanations for each question.\nQuestion: {{input}}\nKnowledge: {{knowledge}}"
    data_template = "{} A. {} B. {}"
elif args.dataset == "obqa":
    prompt_template_part1 = \
    f"The following are multiple-choice questions. Generate a step-by-step explanations for each question.\nQuestion: {{input}}\nKnowledge: {{knowledge}}"
    data_template = "{} A. {} B. {} C. {} D. {}"
else:
    raise NotImplementedError

prompt_template_part2 = "\nExplanation:</s>"

### Path Definition
if args.dataset == "medqa_usmle_hf":
    load_dataset_path = f"data/usmle-cot" # local path to save processed dataset
    save_dataset_path = f"data/usmle-cot-{args.knowledge_base}"
elif args.dataset == "strategyqa":
    load_dataset_path = "data/strategy-cot" # local path to save processed dataset
    save_dataset_path = f"data/strategy-cot-{args.knowledge_base}"
elif args.dataset == "obqa":
    load_dataset_path = "data/obqa-cot" # local path to save processed dataset
    save_dataset_path = f"data/obqa-cot-{args.knowledge_base}"
else
    raise NotImplementedError

if (args.dataset == "obqa" and args.n_knowledge != 10) or (args.dataset != "obqa" and args.n_knowledge != 1):
    save_dataset_path += f"-nknow{args.n_knowledge}"

if args.from_q:
    save_dataset_path += "-fromq"

os.makedirs(save_dataset_path, exist_ok=True)

cprint(f"Save Path: {save_dataset_path}", 'red')

####### Step 1.5 Start --------- 
choices = ["A", "B", "C", "D", "E"]

print("Load Searcher...")
if args.knowledge_base == "wikipedia":
    searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
elif args.knowledge_base == "pubmed":
    searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-bioasq-flat')
else: raise NotImplementedError

with open(os.path.join(load_dataset_path, "train.json"), 'r') as f:
    train_dataset = json.load(f)

with open(os.path.join(load_dataset_path, "test.json"), 'r') as f:
    test_dataset = json.load(f)

new_train_dataset = []
new_test_dataset = []

def parse_knowledge(args, hits):
    if args.knowledge_base == "wikipedia":
        return '\t'.join([hit.raw for hit in hits[:args.n_knowledge]])
    elif args.knowledge_base == "pubmed":
        return json.loads(hits[0].raw)["text"]
    elif args.knowledge_base in ["pileoflaw", "raco"]:
        if len(hits) == 0:
            return ''
        else:
            return '\t'.join([json.loads(hits[i].raw)["contents"] for i in range(args.n_knowledge)])
    else: raise NotImplementedError

q_cache = dict()

if not args.skip_search:
    for data_idx, data in enumerate(tqdm(train_dataset, desc="Append knowledge to train dataset...")):
        question = data["question"]
        reasoning = data["reasoning"]
        answer = data["answer"]

        if "knowledge" not in data.keys():
            if args.from_q:
                q_hash = get_hash_value(question)
                if q_hash in q_cache.keys():
                    hits = q_cache[q_hash]
                else:
                    hits = searcher.search(question)
                    q_cache[q_hash] = hits
            else:
                hits = searcher.search(reasoning) # Retrieve the data with silver reasoning

            knowledge = parse_knowledge(args, hits)
            data["knowledge"] = knowledge
        else:
            knowledge = data["knowledge"]

        new_train_dataset.append(data)

        # if len(new_train_dataset) > 10:
        #     break
        if len(new_train_dataset) < 10:
            cprint("Question", 'red')
            print(question)
            cprint("Reasoning", 'blue')
            print(reasoning)
            cprint("Knowledge", 'green')
            print(knowledge)
            cprint("Reasoning + Answer", 'cyan')
            print(data["reasoning"])
            print()
        
    for data in tqdm(test_dataset, desc="Append knowledge to test dataset..."):
        question = data["question"]
        reasoning = data["reasoning"]

        hits = searcher.search(reasoning) # Retrieve the data with silver reasoning

        knowledge = parse_knowledge(args, hits)
        data["knowledge"] = knowledge

        new_test_dataset.append(data)

        # if len(new_test_dataset) > 10:
        #     break

    # import pdb; pdb.set_trace()
    with open(os.path.join(save_dataset_path, "train.json"), 'w+') as f:
        json.dump(new_train_dataset, f)

    with open(os.path.join(save_dataset_path, "test.json"), 'w+') as f:
        json.dump(new_test_dataset, f)
else:
    exist_flag = os.path.exists(os.path.join(save_dataset_path, "train.json"))
    print(f"Skip Searching... but path exists {exist_flag}")

####### Step 2 Start
dataset = load_dataset('json', data_files={'train': os.path.join(save_dataset_path, "train.json"),
                                          'test': os.path.join(save_dataset_path, "test.json")})

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

max_source_length = tokenizer.model_max_length #### THIS IS NOT HARD LIMIT (We can expand this)

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
target_lengths = [len(x) for x in tokenized_targets["input_ids"]]
# use 95th percentile as max target length
# max_target_length = int(np.percentile(target_lenghts, 95))
max_target_length = max(target_lengths)
print(f"Max target length: {max_target_length}")

def preprocess_function(sample, padding="max_length"):
    # created prompted input
    inputs = [prompt_template_part1.format(input=item, knowledge=knowledge) for item, knowledge in zip(sample[text_column], sample[knowledge_column])]

    inputs2 = [prompt_template_part2 for item in sample[text_column]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, inputs2, max_length=max_source_length, 
                             padding=padding, truncation=True, add_special_tokens=False)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample[summary_column], max_length=max_target_length, padding=padding, 
                       truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# process dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))

# save dataset to disk
tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path, "train"))
tokenized_dataset["test"].save_to_disk(os.path.join(save_dataset_path, "eval"))