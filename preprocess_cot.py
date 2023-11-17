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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="medqa_usmle_hf", 
    choices=["medqa_usmle_hf", "strategyqa", "obqa"])
args = parser.parse_args()

### Experiment config
model_id = "google/flan-t5-large" # Hugging Face Model Id
text_column = "question" # column of input text is
summary_column = "reasoning" # column of the output text
if args.dataset == "medqa_usmle_hf":
    prompt_template = f"The following are multiple-choice questions about medical knowledge. Generate a step-by-step explanations for each question:\n{{input}}\nExplanation:\n"
    data_template = "{} A. {} B. {} C. {} D. {}"
elif args.dataset == "strategyqa":
    prompt_template = f"The following are multiple-choice questions. Generate a step-by-step explanations for each question:\n{{input}}\nExplanation:\n"
    data_template = "{} A. {} B. {}"
elif args.dataset == "obqa":
    prompt_template = f"The following are multiple-choice questions about commonsense knowledge. Generate a step-by-step explanations for each question:\n{{input}}\nExplanation:\n"
    data_template = "{} A. {} B. {} C. {} D. {}"
else:
    raise NotImplementedError

def format_question(args, data):
    if args.dataset in ["medqa_usmle_hf", "obqa"]:
        q = data_template.format(data["sent1"], data["ending0"], data["ending1"], data["ending2"], data["ending3"])
    elif args.dataset in ["strategyqa"]:
        q = data_template.format(data["sent1"], data["ending0"], data["ending1"])
    else:
        raise NotImplementedError
    return q

choices = ["A", "B", "C", "D"]

### Path Definition
if args.dataset == "medqa_usmle_hf":
    save_dataset_path = "preprocessed_data/medqa-cot" # local path to save processed dataset
elif args.dataset == "strategyqa":
    save_dataset_path = "preprocessed_data/strategyqa-cot"
elif args.dataset == "obqa":
    save_dataset_path = "preprocessed_data/obqa-cot"
else:
    raise NotImplementedError
os.makedirs(save_dataset_path, exist_ok=True)

cot_dump_path = f"./chain_of_thoughts/{args.dataset}"
os.makedirs(cot_dump_path, exist_ok=True)
fpx = open(os.path.join(cot_dump_path, "train_dump.jsonl"), 'w+')

### CoT Path
cot_dir_path = f"./chatgpt_output/{args.dataset}/train"

print(args.dataset)

# Load raw dataset
dataset_raw = load_dataset("json", data_files=f"data/{args.dataset}/train.json", split="train")

verifier = Verifier(args)

new_dataset = []

cot_dumps = []

for idx, data in enumerate(tqdm(dataset_raw)):
    cot_file_path = os.path.join(cot_dir_path, str(idx).zfill(5) + ".json")

    if not os.path.exists(cot_file_path): 
        print(f"Not Exist {cot_file_path}")
        print("We expert there is no more cot afterwards in now....")
        break

    with open(cot_file_path) as f:
        cot_data = json.load(f)

    correct_cot, wrong_cot = verifier(data, cot_data)

    cot_data = {
        "correct": correct_cot,
        "wrong": wrong_cot,
    }
    # cot_dumps.append(cot_data)
    fpx.write(json.dumps(cot_data))
    fpx.write('\n')

    for cot in correct_cot:
        answer = data["label"]
        option = choices[int(answer)]
        answer_prompt = f"\nAnswer: {option}"
        new_data = {
            "question": format_question(args, data),
            "reasoning": cot + answer_prompt,
            "answer": data["label"]
        }
        new_dataset.append(new_data)

        if len(new_dataset) < 10:
            cprint("Question", 'red')
            print(new_data["question"])
            cprint("Reasoning", 'blue')
            print(new_data["reasoning"])
            print("+" * 20)
            print()

    # if len(new_dataset) > 100:
    #     break
fpx.close()

print(f"New Dataset Size: {len(new_dataset)}")
train_len = int(len(new_dataset) * 0.95)
train_dataset = new_dataset[:train_len]
test_dataset = new_dataset[train_len:] # Split validation set for Reasoning Distillation

with open(os.path.join(save_dataset_path, "train.json"), 'w+') as f:
    json.dump(train_dataset, f)

with open(os.path.join(save_dataset_path, "test.json"), 'w+') as f:
    json.dump(test_dataset, f)


####### Step 1 done
dataset = load_dataset('json', data_files={'train': os.path.join(save_dataset_path, "train.json"),
                                          'test': os.path.join(save_dataset_path, "test.json")})

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

prompt_lenght = len(tokenizer(prompt_template.format(input=""))["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_lenght
print(f"Prompt length: {prompt_lenght}")
print(f"Max input length: {max_sample_length}")

tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[text_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_sample_length)
print(f"Max source length: {max_source_length}")

tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
target_lengths = [len(x) for x in tokenized_targets["input_ids"]]
max_target_length = int(np.max(target_lengths))
print(f"Max target length: {max_target_length}")


def preprocess_function(sample, padding="max_length"):
    # created prompted input
    inputs = [prompt_template.format(input=item) for item in sample[text_column]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample[summary_column], max_length=max_target_length, padding=padding, truncation=True)

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