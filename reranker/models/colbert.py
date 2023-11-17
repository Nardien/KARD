from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
import numpy as np
from einops import rearrange

class ColBERTConfig(PretrainedConfig):
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True

class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    We use a dot-product instead of cosine per term (slightly better)
    """
    config_class = ColBERTConfig
    base_model_prefix = "bert_model"

    def __init__(self, cfg, n_cands=8, update_both=False) -> None:
        super().__init__(cfg)
        
        print(f"Inside the ColBERT: {cfg._name_or_path}")
        self.bert = AutoModel.from_pretrained(cfg._name_or_path)

        # for p in self.bert.parameters():
        #     p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert.config.hidden_size, cfg.compression_dim)

        self.n_cands = n_cands
        self.update_both = update_both
        print(f"Model n_cands: {self.n_cands}")

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor]):

        query_vecs = self.forward_representation(query)
        document_vecs = self.forward_representation(document, sequence_type="doc")

        score = self.forward_aggregation(query_vecs, document_vecs, query["attention_mask"], document["attention_mask"])
        return score

    def forward_representation(self,
                               tokens,
                               sequence_type=None) -> torch.Tensor:
        
        if sequence_type == "doc":
            if self.update_both:
                vecs = self.bert(**tokens)[0]
            else:
                with torch.no_grad():
                    vecs = self.bert(**tokens)[0] # assuming a distilbert model here
        else:
            vecs = self.bert(**tokens)[0]
        vecs = self.compressor(vecs)
        return vecs

    def forward_aggregation(self, query_vecs, document_vecs, query_mask, document_mask):
        # query_vecs: B x N x D
        # doc_vecs: (B * k) x N x D

        # Unsqueeze query vector
        _bsz = query_vecs.shape[0]
        n_cands = document_vecs.shape[0] // _bsz
        query_vecs_dup = query_vecs.repeat_interleave(n_cands, dim=0).contiguous()

        score = torch.bmm(query_vecs_dup, document_vecs.transpose(1, 2))
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~exp_mask] = - 10000

        # max pooling over document dimension
        score = score.max(-1).values
        query_mask_dup = query_mask.repeat_interleave(n_cands, dim=0).contiguous()

        score[~(query_mask_dup.bool())] = 0
        score = rearrange(score.sum(-1), '(b n) -> b n', n=n_cands) # B x k 
        return score

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
    model = ColBERT.from_pretrained("michiyasunaga/BioLinkBERT-base")

    query = ["Pressure reactivity index or PRx is tool for monitoring patients who have raised intracranial pressure (ICP)",
             "monitoring patients"]
    keys = [
        "caused by pathologies such as a traumatic brain injury or subarachnoid haemorrhage",
    ] * 8 + [
        "in order to guide therapy to protect the brain from damagingly high or low cerebral blood flow."
    ] * 8

    model.to("cuda")

    max_seq_len = 512

    query_outputs = tokenizer(query, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)
    key_outputs = tokenizer(keys, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)

    query_outputs = {k: v.to("cuda") for k, v in query_outputs.items()}
    key_outputs = {k: v.to("cuda") for k, v in key_outputs.items()}

    model(query_outputs, key_outputs)
