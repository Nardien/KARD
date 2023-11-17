# Reranker Guide

## How to train?

```bash
python main.py --dataset {medqa_usmle_hf,strategyqa,obqa} --generate_search_space
```

For training, we cache the top-k candidate documents from the retriever as the question or the rationale as a query.
Based on your infrastructure, the size of dataset, and the size of knowledge base, this process can take too long time to build the cache.
For faster setup of experiments, we provide the pre-built cache in [this link](https://drive.google.com/file/d/175IgrCiq4Bi6u-0O8wc14XspWv-y5t6k/view?usp=drive_link).

Instead of training reranker, you can download the best reranker checkpoint we used in the main experiments for each dataset from [this link](https://drive.google.com/file/d/13pqMwFsLdkZrjT9F55duSS9Tnb36Z8U1/view?usp=sharing).