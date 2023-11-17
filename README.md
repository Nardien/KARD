# KARD: Knowledge-Augmented Reasoning Distillation

Official Code Repository for the paper [Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-intensive Tasks](https://arxiv.org/abs/2305.18395) (NeurIPS 2023).

In this repository, we initially provide the trainig code with our KARD method.


## Abstract
<img align="middle" width="900" src="https://github.com/Nardien/KARD_test/blob/main/images/concept_figure.PNG">

Large Language Models (LLMs) have shown promising performance in knowledge-intensive reasoning tasks that require a compound understanding of knowledge. However, deployment of the LLMs in real-world applications can be challenging due to their high computational requirements and concerns on data privacy. Previous studies have focused on building task-specific small Language Models (LMs) by fine-tuning them with labeled data or distilling LLMs. However, these approaches are ill-suited for knowledge-intensive reasoning tasks due to the limited capacity of small LMs in memorizing the knowledge required. Motivated by our theoretical analysis on memorization, we propose Knowledge-Augmented Reasoning Distillation (KARD), a novel method that fine-tunes small LMs to generate rationales obtained from LLMs with augmented knowledge retrieved from an external knowledge base. Moreover, we further propose a neural reranker to obtain documents relevant to rationale generation. We empirically show that KARD significantly improves the performance of small T5 and GPT models on the challenging knowledge-intensive reasoning datasets, namely MedQA-USMLE, StrategyQA, and OpenbookQA. Notably, our method makes the 250M T5 models achieve superior performance against the fine-tuned 3B models, having 12 times larger parameters, on both MedQA-USMLE and StrategyQA benchmarks.

## Installation
Python version: 3.8.0
```bash
python -m pip install -r requirements.txt
```

## Dataset
Follow below steps to setup the dataset for training.

1. Download the raw dataset from [this link](https://drive.google.com/file/d/16Niskw2zcvyIdeRUEB2yjU2QQFy2wN3W/view?usp=share_link).
2. Download the rationales from the teacher language model (ChatGPT) from [this link](https://drive.google.com/file/d/1tRBSLf9BeRyrsK4g2M-lRTR56OuJCt6E/view?usp=sharing).
3. Run `python preprocess_cot.py --dataset {medqa_usmle_hf,strategyqa,obqa}`
4. Run `python preprocess_cot_grounded.py --dataset {medqa_usmle_hf,strategyqa,obqa} --knowledge_base {wikipedia,pubmed}`

You can also download the preprocessed data from [this link](https://drive.google.com/file/d/118rvsqpTIHjoOuNgeYmyh7PrlmoKeAm3/view?usp=sharing).

## LM Training Guide
If you want to run Knowledge-Augmented Reasoning Distillation, run the below script:

```bash
sh scripts/run_kard.sh {GPU ID} {Batch Size per GPU} {Model Size:base,large,xl} {Dataset:medqa,strategyqa,obqa}
```

We also provide the script for Reasoning Distillation without knowledge augmnetation.

```bash
sh scripts/run_rd.sh {GPU ID} {Batch Size per GPU} {Model Size:base,large,xl} {Dataset:medqa,strategyqa,obqa}
```

Both training script supports multi-gpu training.

For example, if you want to run KARD on the xl-sized LM training on medqa dataset with 4 gpus with batch size 8 per GPU, run as follows:

```bash
sh scripts/run_kard.sh 0,1,2,3 8 xl medqa
```

## Reranker Training Guide
To train the reranker, check the `reranker` folder.


## Inference Guide
After the LM and reranker training, run the following code:
```bash
python generate_predict.py --checkpoint_path "/path/to/checkpoint/" --retriever_type {sparse,dense} --dataset {medqa_usmle_hf,strategyqa,obqa} --dense_retriever_path "/path/to/retriever/"
```

If you train both the LM and reranker following the code above, run the code as follows:
```bash
python generate_predict.py --checkpoint_path "./save/flan-t5-base/medqa/kard_wikipedia" --retriever_type dense --dataset medqa_usmle_hf --dense_retriever_path "./reranker/save/colbert_lr1e-4"
```

You can adjust the following hyperparmeters:
- `--max_doc`: adjust the number of maximum passages in the candidate set for reranking.
- `--n_docs`: the number of passages to be appended into the prompt

## TODOs
- [ ]  Add codes for GPT.

Please feel free to leave it on Issues if you find any problems.


## Citation
If you found this work useful, please cite our work.
```
@inproceedings{DBLP:conf/nips/KangLBKS23,
  author       = {Minki Kang and
                  Seanie Lee and
                  Jinheon Baek and
                  Kenji Kawaguchi and
                  Sung Ju Hwang},
  title        = {Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023, December
                  10-16, 2023, New Orleans},
  year         = {2023},
}
```
