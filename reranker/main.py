import argparse
import os
import datetime
from tqdm import tqdm
import numpy as np
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

from retriever_dataset import RetrieverDataset, batch_collate
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from models.colbert import ColBERT

from accelerate import Accelerator

from utils import set_seed

import wandb

from retriever_dataset import RetrieverDataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def run(args):
    if not args.no_report:
        wandb.init(project="KARD-reranker")
        wandb.config.update(args)
        wandb.run.name = os.path.basename(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ColBERT.from_pretrained(args.model_name, n_cands=args.n_cands)

    peft_config = LoraConfig(
        target_modules=["query", "value"], 
        inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1,
        modules_to_save=["compressor"]
    )
    peft_config.save_pretrained(os.path.join("save", args.save_dir, "model"))
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    device = "cuda:0"
    model.to(device)
    model.train()

    train_dataset = RetrieverDataset(args, tokenizer, fold="train")
    eval_dataset = RetrieverDataset(args, tokenizer, fold="eval")

    bsz = args.batch_size
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=bsz, collate_fn=batch_collate,)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_loader = DataLoader(dataset=eval_dataset, sampler=eval_sampler, batch_size=bsz, collate_fn=batch_collate,)

    accelerator = Accelerator(mixed_precision='fp16')

    if args.loss_type == "kl":
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
    elif args.loss_type == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)

    ### Hyperparameters
    tau = args.tau
    tau2 = args.model_tau

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    pbar = tqdm(total=args.num_epochs * len(train_loader), desc="Training...")
    current_step = 0
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            model.zero_grad()
            query_batch = {k.replace("query_", ""): v.to(device) for k, v in batch.items() if "query_" in k}
            key_batch = {k.replace("key_", ""): v.to(device) for k, v in batch.items() if "key_" in k}

            similarity = model(query_batch, key_batch)

            if args.loss_type == "kl":
                pred_dist = torch.log_softmax(similarity / tau2, dim=-1)
                gold_dist = torch.softmax(batch["label"].to(device) / tau, dim=-1)

                if args.tau_annealing:
                    alpha = 2 / (1 + np.exp(-10 * min(current_step, total_steps) / total_steps)) - 1
                    tau = args.tau - alpha * 4.0
                    tau2 = args.model_tau - alpha * 400.0

                loss = criterion(pred_dist, gold_dist)
            elif args.loss_type == "ce":
                pred_dist = torch.log_softmax(similarity / tau2, dim=-1)
                gold_dist = torch.softmax(batch["label"].to(device) / tau, dim=-1)

                label = batch["label"].argmax(dim=-1)
                loss = criterion(similarity / tau2, label)

            loss = loss * args.alpha

            pred_entropy = torch.distributions.Categorical(torch.exp(pred_dist)).entropy().mean()
            gold_entropy = torch.distributions.Categorical(gold_dist).entropy().mean()

            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            if not args.no_scheduler:
                scheduler.step()
            pbar.update()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.2f}",
                }
            )

            if not args.no_report:
                wandb.log(
                    {
                        "Train/loss": loss.item(),
                        "Train/gold_entropy": gold_entropy.item(),
                        "Train/pred_entropy": pred_entropy.item(),
                        "Train/tau": tau,
                        "Train/tau2": tau2,
                    },
                    step=current_step
                )
            current_step += 1

        if not args.debug:
            acc, acc_top3, acc_top5, eval_loss = evaluate(args, model, eval_loader, device)
            print(f"Epoch [{epoch}/{args.num_epochs}]: Accuracy Top1 {acc:.2f} Top3 {acc_top3:.2f} Top5 {acc_top5:.2f}")
            if not args.no_report:
                wandb.log(
                    {
                        "Eval/accuracy_top1": acc,
                        "Eval/accuracy_top3": acc_top3,
                        "Eval/accuracy_top5": acc_top5,
                        "Eval/loss": eval_loss,
                    },
                    step=current_step
                )

        model.save_pretrained(os.path.join("save", args.save_dir, "model"))
        tokenizer.save_pretrained(os.path.join("save", args.save_dir))
        
def evaluate(args, model, dataloader, device, tau=500):
    bsz = args.batch_size

    n_total = 0
    n_correct = 0
    n_correct_top3 = 0
    n_correct_top5 = 0

    model.eval()

    tau = args.tau
    tau2 = args.model_tau
    eval_loss = 0

    for batch in tqdm(dataloader, desc="Evaluate..."):
        query_batch = {k.replace("query_", ""): v.to(device) for k, v in batch.items() if "query_" in k}
        key_batch = {k.replace("key_", ""): v.to(device) for k, v in batch.items() if "key_" in k}
        _bsz = query_batch["input_ids"].shape[0]
        
        with torch.no_grad():
            similarity = model(query_batch, key_batch)

        pred_dist = torch.log_softmax(similarity / tau2, dim=-1)
        gold_dist = torch.softmax(batch["label"].to(device) / tau, dim=-1)
        eval_loss += F.kl_div(pred_dist, gold_dist, reduction='none').mean(-1).sum().item()

        preds = pred_dist.argsort(dim=-1, descending=True)

        for pred in preds: # Iteration over batch
            if 0 in pred[:1]: n_correct += 1
            if 0 in pred[:3]: n_correct_top3 += 1
            if 0 in pred[:5]: n_correct_top5 += 1

        n_total += preds.shape[0]

    acc = n_correct / n_total * 100
    acc_top3 = n_correct_top3 / n_total * 100
    acc_top5 = n_correct_top5 / n_total * 100

    eval_loss = eval_loss / n_total
    model.train()
    print(n_correct)
    print(n_correct_top3)
    print(n_correct_top5)
    print(n_total)
    print(eval_loss)
    return acc, acc_top3, acc_top5, eval_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="medqa_usmle_hf", choices=["medqa_usmle_hf", "strategyqa", "obqa"])
    parser.add_argument("--data_dir", type=str, default="../Biomedical_CoT_Generation/data/usmle-cot") # Auto-Set
    parser.add_argument("--search_space_dir", type=str, default="./search_spaces") # Auto-Set
    parser.add_argument("--knowledge_base", type=str, default="wikipedia", choices=["wikipedia", "pubmed"])
    parser.add_argument("--save_dir", type=str, default="colbert_lr1e-3")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_cands", type=int, default=8, help="The number of negative samples")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--model_tau", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_report", action='store_true')
    parser.add_argument("--tau_annealing", action='store_true')
    parser.add_argument("--update_both", action='store_true')
    parser.add_argument("--loss_type", type=str, default="ce", choices=["kl", "ce"])
    parser.add_argument("--generate_search_space", action='store_true')
    parser.add_argument("--no_scheduler", action='store_true')
    parser.add_argument("--alpha", default=1.0, type=float)

    parser.add_argument("--only_from_rationale", action='store_true', help="coupled with combine candidates, but use candidate from r only")
    parser.add_argument("--only_from_question", action='store_true', help="coupled with combine candidates, but use candidate from q only")

    args = parser.parse_args()

    if args.dataset == "medqa_usmle_hf":
        args.data_dir = "../preprocessed_data/medqa-cot"
        if args.model_name is None:
            args.model_name = "michiyasunaga/BioLinkBERT-base"
    elif args.dataset == "strategyqa":
        args.data_dir = "../preprocessed_data/strategyqa-cot"
        if args.model_name is None:
            args.model_name = "michiyasunaga/LinkBERT-base"
    elif args.dataset == "obqa":
        args.data_dir = "../preprocessed_data/obqa-cot"
        if args.model_name is None:
            args.model_name = "michiyasunaga/LinkBERT-base"
    else:
        raise NotImplementedError

    args.search_space_dir = os.path.join(args.search_space_dir, f"{args.dataset}-{args.knowledge_base}")
    print(f"Search Space: {args.search_space_dir}")

    args.tau_annealing = True
    args.update_both = True
    print(f"  [Default Setting] tau_annealing {args.tau_annealing}")
    print(f"  [Default Setting] update parameters from both query and key {args.update_both}")

    set_seed(42)
    run(args)