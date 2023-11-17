import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig
from peft import PeftModel, PeftConfig
from einops import rearrange
from typing import Dict
from termcolor import cprint

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class Retriever:
    def __init__(self, args):

        self.query_model_path = args.dense_retriever_path
        self.retriever_model = args.retriever_model

        # Forced change
        self.retriever_model = "colbert"
        self.device = args.device

        if self.query_model_path is None or args.retriever_type == "sparse":
            self.query_model, self.key_model, self.tokenizer = None, None, None
            self.max_seq_len = 512
        elif self.retriever_model == "colbert":
            _config = PeftConfig.from_pretrained(self.query_model_path)
            base_model = _config.base_model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

            self.query_model = ColBERT.from_pretrained(base_model)
            self.query_model = PeftModel.from_pretrained(self.query_model, self.query_model_path)
            self.query_model.eval()
            self.query_model.to(self.device)
            self.max_seq_len = min(self.tokenizer.model_max_length, 512)
        else:
            raise NotImplementedError


    def retrieve(self, q, documents, n=1, n_cands=-1, return_rank=False):
        if self.query_model is None: return documents[0]

        if self.retriever_model in ["colbert"]:
            query_outputs = self.tokenizer(q, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)
            key_outputs = self.tokenizer(documents, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)

            query_inputs = {k:v.to(self.device) for k, v in query_outputs.items()}
            key_inputs = {k:v.to(self.device) for k, v in key_outputs.items()}

            with torch.no_grad():
                similarity = self.query_model(query_inputs, key_inputs)
            scores = similarity.flatten().tolist()
            ranking = sorted([(idx, score) for idx, score in enumerate(scores)], key=lambda x: x[1], reverse=True)
            documents = sorted([(doc, score) for doc, score in zip(documents, scores)], key=lambda x: x[1], reverse=True)
            documents = [doc[0] for doc in documents]

        else:
            raise NotImplementedError

        if return_rank: return documents, ranking
        return documents

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

    def __init__(self, cfg, n_cands=8) -> None:
        super().__init__(cfg)
        
        self.bert = AutoModel.from_pretrained(cfg._name_or_path)

        # for p in self.bert.parameters():
        #     p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert.config.hidden_size, cfg.compression_dim)

        self.n_cands = n_cands
        print(f"Model n_cands: {self.n_cands}")

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor]):

        query_vecs = self.forward_representation(query)
        document_vecs = self.forward_representation(document)

        score = self.forward_aggregation(query_vecs, document_vecs, query["attention_mask"], document["attention_mask"])
        return score

    def forward_representation(self,
                               tokens,
                               sequence_type=None) -> torch.Tensor:
        
        vecs = self.bert(**tokens)[0] # assuming a distilbert model here
        vecs = self.compressor(vecs)

        # # if encoding only, zero-out the mask values so we can compress storage
        # if sequence_type == "doc_encode" or sequence_type == "query_encode": 
        #     vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)

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