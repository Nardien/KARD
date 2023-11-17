import json
import os
import sys
sys.path.append('.')
import random
import argparse
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
from vectorizer import BM25Vectorizer

import hashlib
def get_hash_value(in_str, in_digest_bytes_size=64, in_return_type='hexdigest'):
    assert 1 <= in_digest_bytes_size and in_digest_bytes_size <= 64
    blake  = hashlib.blake2b(in_str.encode('utf-8'), digest_size=in_digest_bytes_size)
    if in_return_type == 'hexdigest': return blake.hexdigest()
    elif in_return_type == 'number': return int(blake.hexdigest(), base=16)
    return blake.digest()

class RetrieverDataset(Dataset):
    def __init__(self, args, tokenizer, fold="train"):
        self.args = args
        with open(os.path.join(args.data_dir, "train.json"), 'r') as f:
            self.original_dataset = json.load(f)

        # Debug
        if args.debug:
            self.original_dataset = self.original_dataset[:100]

        self.dataset = self.split_set(self.original_dataset, fold, 0.95)
        ## Labeling data
        self.dataset = [(_id, data) for _id, data in enumerate(self.dataset)]

        self.tokenizer = tokenizer
        self.max_seq_len = 512

        if args.knowledge_base == "wikipedia":
            print("Load Searcher...")
            self.searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
            print("Load Index...")
            self.index_reader = IndexReader.from_prebuilt_index('enwiki-paragraphs')
            self.vectorizer = BM25Vectorizer('enwiki-paragraphs')
        elif args.knowledge_base == "pubmed":
            print("Load Searcher...")
            self.searcher = LuceneSearcher('beir-v1.0.0-bioasq-flat')
            print("Load Index...")
            self.index_reader = IndexReader('beir-v1.0.0-bioasq-flat')
            self.vectorizer = BM25Vectorizer('beir-v1.0.0-bioasq-flat')

        self.fold = fold
        self.k = args.n_cands if fold == "train" else 8
        self.n_k = 32
        
        os.makedirs(args.search_space_dir, exist_ok=True)
        self.search_space_dump = os.path.join(args.search_space_dir, f"search_space_{fold}")
        os.makedirs(self.search_space_dump, exist_ok=True)
        self.generate_search_space = args.generate_search_space

        self.search()

    def parse_document(self, hit):
        if self.args.knowledge_base == "wikipedia":
            return hit.raw
        elif self.args.knowledge_base in ["pileoflaw", "raco"]:
            return json.loads(hit.raw)["contents"]
        elif self.args.knowledge_base in ["pubmed"]:
            return json.loads(hit.raw)["text"]
        
    def search(self):
        print(self.search_space_dump)
        if not self.generate_search_space:
            print("Dump Exists.")
        else:
            q_retrieval_cache = dict()
            for _data in tqdm(self.dataset, desc="Preprocess dataset.."):
                idx, data = _data
                dump_path = os.path.join(self.search_space_dump, str(idx).zfill(7) + ".json")
                if os.path.exists(dump_path):
                    continue

                question = data["question"]
                reasoning = data["reasoning"]

                silver_hits = self.searcher.search(reasoning, k=self.n_k)

                # load 
                r_search_space = []
                for i, hit in enumerate(silver_hits):
                    score = hit.score
                    document = self.parse_document(hit)
                    r_search_space.append({"docid": hit.docid, "document": document, "score": score})

                    if i != 0: continue
                    print(reasoning)
                    print(document)
                    print()                

                q_search_space = []
                q_hash = get_hash_value(question)
                if q_hash in q_retrieval_cache:
                    silver_hits = q_retrieval_cache[q_hash]
                else:
                    silver_hits = self.searcher.search(question, k=self.n_k)
                    q_retrieval_cache[q_hash] = silver_hits

                for i, hit in enumerate(silver_hits):
                    score = hit.score
                    document = self.parse_document(hit)
                    
                    # Compute similarity again
                    q_search_space.append({"docid": hit.docid, "document": document, "score": score})

                q_search_space = self.recompute_score(reasoning, q_search_space)

                search_space = {
                    'q': q_search_space,
                    'r': r_search_space,
                }

                with open(dump_path, 'w') as f:
                    json.dump(search_space, f, indent=4)
    
    def recompute_score(self, q, search_space):
        q_vector = self.vectorizer.get_query_vector(q)
        doc_vectors = self.vectorizer.get_vectors([d["docid"] for d in search_space])

        scores = np.asarray(q_vector.dot(doc_vectors.transpose()).todense())[0].tolist()
        for item, new_score in zip(search_space, scores):
            item["score"] = new_score

        return search_space

    def split_set(self, dataset, fold, ratio=0.95):
        datalen = len(dataset)
        if fold == "train":
            dataset = dataset[:int(ratio * datalen)]
        else:
            dataset = dataset[int(ratio * datalen):]
        return dataset

    def preprocess(self, data, space):
        query = data["question"]
        keys = [item["document"] for item in space]

        query_outputs = self.tokenizer(query, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)
        key_outputs = self.tokenizer(keys, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)

        scores = [item["score"] for item in space]

        query_input_ids = query_outputs["input_ids"]
        query_attention_mask = query_outputs["attention_mask"]
        key_input_ids = key_outputs["input_ids"]
        key_attention_mask = key_outputs["attention_mask"]
        score = torch.FloatTensor(scores)

        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "key_input_ids": key_input_ids,
            "key_attention_mask": key_attention_mask,
            "label": score,
        }

    def __getitem__(self, idx):
        _idx, data = self.dataset[idx]
        load_path = os.path.join(self.search_space_dump, str(_idx).zfill(7) + ".json")
        with open(load_path, 'r') as f:
            _ss = json.load(f)
  
        if self.args.only_from_rationale:
            ss = _ss['r']
        elif self.args.only_from_question:
            ss = _ss['q']
        else:
            ss = _ss['r'] + _ss['q']

        if self.k < len(ss):
            if self.fold == "train":
                if self.args.only_from_rationale:
                    ss = random.sample(_ss['r'], self.k)
                elif self.args.only_from_question:
                    ss = random.sample(_ss['q'], self.k)
                else:
                    ss = random.sample(_ss['r'], self.k // 2) + random.sample(_ss['q'], self.k // 2)
            else:
                ss = _ss['r'] + _ss['q']
                _ss = random.sample(ss[1:], self.k - 1)
                ss = [ss[0]] + _ss
        return self.preprocess(data, ss)

    def __len__(self):
        return len(self.dataset)


def batch_collate(batch):
    batch_query_input_ids = torch.cat([item["query_input_ids"] for item in batch], dim=0)
    batch_query_attention_mask = torch.cat([item["query_attention_mask"] for item in batch], dim=0)
    batch_key_input_ids = torch.cat([item["key_input_ids"] for item in batch], dim=0)
    batch_key_attention_mask = torch.cat([item["key_attention_mask"] for item in batch], dim=0)
    batch_label = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "query_input_ids": batch_query_input_ids,
        "query_attention_mask": batch_query_attention_mask,
        "key_input_ids": batch_key_input_ids,
        "key_attention_mask": batch_key_attention_mask,
        "label": batch_label,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../Biomedical_CoT_Generation/data/usmle-cot")
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--n_negative", type=int, default=6, help="The number of negative samples")
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    RetrieverDataset(args, tokenizer)