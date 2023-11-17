import random
import os
import json
import argparse
from tqdm import tqdm
from termcolor import cprint
from pyserini.search.lucene import LuceneSearcher

# Deterministic
def do_cache(args):
    dataset_path = f"./data/{args.dataset}"
    filename = f"{args.fold}.json"
    with open(os.path.join(dataset_path, filename)) as f:
        dataset = [json.loads(data) for data in f.readlines()]
    output_dir = f"./retriever_cache/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    if args.knowledge_base == "wikipedia":
        searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
    elif args.knowledge_base == "pubmed":
        searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-bioasq-flat')

    if args.dataset == "strategyqa":
        data_template = "{} A. {} B. {}"
    else:
        data_template = "{} A. {} B. {} C. {} D. {}"

    # Debug
    # dataset = dataset[:100]
    
    chatgpt_cot_dataset = []
    if args.ground_truth:
        chatgpt_path = f"./chatgpt_output/{args.dataset}/test/"
        for i in range(len(dataset)):
            with open(os.path.join(chatgpt_path, str(i).zfill(5) + ".json"), 'r') as f:
                data = json.load(f)
                chatgpt_cot = data["choices"][0]["message"]["content"]
                chatgpt_cot_dataset.append(chatgpt_cot)
        k = 10
    else:
        k = 300 if args.max_doc < 0 else args.max_doc
    print(f"k = {k}")

    results = []
    for idx, data in enumerate(tqdm(dataset)):
        if args.ground_truth or args.keywords or args.ground_truth_keywords:
            q = chatgpt_cot_dataset[idx]
            if args.dataset == "strategyqa":
                data = data_template.format(data["sent1"], data["ending0"], data["ending1"])
            else:
                data = data_template.format(data["sent1"], data["ending0"], data["ending1"], data["ending2"], data["ending3"])
        else:
            if args.dataset == "strategyqa":
                q = data_template.format(data["sent1"], data["ending0"], data["ending1"])
            else:
                q = data_template.format(data["sent1"], data["ending0"], data["ending1"], data["ending2"], data["ending3"])
            data = ''
        documents = [hit.docid for hit in searcher.search(q, k=k)]
        results.append({'id': idx, 'documents': documents})

        if idx < 10:
            cprint("Question", 'red')
            print(data)
            cprint("Reasoning", 'blue')
            print(q)
            cprint("Top1 Knowledge", 'green')
            if args.knowledge_base == "wikipedia":
                print(searcher.doc(documents[0]).raw())
            elif args.knowledge_base == "pubmed":
                print(json.loads(searcher.doc(documents[0]).raw())["text"])
            print()

    if args.ground_truth:
        output_filename = os.path.join(output_dir, f"{args.fold}_{args.knowledge_base}_GT_cache.jsonl")
    else:
        output_filename = os.path.join(output_dir, f"{args.fold}_{args.knowledge_base}_cache.jsonl")
    with open(output_filename, 'w') as f:
        for result in results:
            f.write(json.dumps(result))
            f.write('\n')

    print(f"Done {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_base', type=str, default="wikipedia", choices=[None, "wikipedia", "pubmed"])
    parser.add_argument('--fold', type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument('--ground_truth', action='store_true')
    parser.add_argument('--ground_truth_keywords', action='store_true')
    parser.add_argument('--keywords', action='store_true')
    parser.add_argument('--dataset', type=str, default="medqa_usmle_hf", choices=["medqa_usmle_hf", "strategyqa", "obqa"])
    parser.add_argument('--from_short', action='store_true', help="Indicate whether it is from short rationale or not -- only for GT")
    parser.add_argument('--max_doc', type=int, default=-1)
    args = parser.parse_args()
    do_cache(args)