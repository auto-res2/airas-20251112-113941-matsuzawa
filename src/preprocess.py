import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import datasets
import numpy as np
from datasets import load_dataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

__all__ = ["build_dataloaders", "SentinelPool"]

# --------------------------------------------------------------------------------
#  COLLATE FUNCTION
# --------------------------------------------------------------------------------

def _make_collate(tokenizer):
    def _collate(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        out = tokenizer.pad({"input_ids": input_ids, "labels": labels}, padding="longest", return_tensors="pt")
        out["attention_mask"] = (out["input_ids"] != tokenizer.pad_token_id).long()
        out["prompt_text"] = [item["prompt_text"] for item in batch]
        out["answer_text"] = [item["answer_text"] for item in batch]
        return out
    return _collate

# --------------------------------------------------------------------------------
#  GSM-8K PREPROCESSING
# --------------------------------------------------------------------------------

def _prep_examples(examples, tok, max_len):
    res = {"input_ids": [], "labels": [], "prompt_text": [], "answer_text": []}
    for q, a in zip(examples["question"], examples["answer"]):
        prompt = q.strip() + "\n\nAnswer:"
        answer_str = a.strip()
        p_ids = tok(prompt, truncation=True, max_length=max_len).input_ids
        a_ids = tok(" " + answer_str + tok.eos_token, truncation=True, max_length=max_len).input_ids
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        res["input_ids"].append(ids)
        res["labels"].append(labels)
        res["prompt_text"].append(prompt)
        res["answer_text"].append(answer_str)
    return res

# --------------------------------------------------------------------------------
#  PUBLIC API: BUILD DATALOADERS
# --------------------------------------------------------------------------------

def build_dataloaders(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    if cfg.dataset.name.lower() == "gsm8k":
        ds = load_dataset("openai/gsm8k", cfg.dataset.config, cache_dir=".cache")
        train_ds = ds["train"].map(lambda ex: _prep_examples(ex, tokenizer, cfg.dataset.max_seq_length), batched=True,
                                    remove_columns=ds["train"].column_names)
        dev_ds = ds["test"].map(lambda ex: _prep_examples(ex, tokenizer, cfg.dataset.max_seq_length), batched=True,
                                  remove_columns=ds["test"].column_names)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset.name}")

    collate = _make_collate(tokenizer)
    train_loader = DataLoader(train_ds.shuffle(seed=cfg.seed), batch_size=cfg.dataset.batch_size, shuffle=True,
                              num_workers=cfg.dataset.num_workers, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=min(8, cfg.dataset.batch_size), shuffle=False,
                            num_workers=cfg.dataset.num_workers, collate_fn=collate)
    return train_loader, dev_loader, tokenizer

# --------------------------------------------------------------------------------
#  SENTINEL POOL (capability-specific prompt mining)
# --------------------------------------------------------------------------------

_CAP_PATTERNS = {
    "syntax": [r"\b(is|are|was|were|the|a)\b"],
    "facts": [r"Who", r"When", r"Where", r"Which"],
    "arith": [r"How many", r"What is", r"Calculate"],
    "logic": [r"If", r"then", r"must", r"always"],
    "global": [r"."]  # fallback pattern
}


class SentinelPool:
    """Maintains a capability-partitioned pool of sentinel prompts refreshed from
    a streaming web corpus (C4)."""

    def __init__(self, tokenizer, taxonomy: List[str], pool_size_per_cap: int, refresh_interval: int,
                 stream_cfg: Dict[str, Any], device):
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.pool_size = pool_size_per_cap
        self.refresh_interval = refresh_interval
        self.device = device
        self.stream_cfg = stream_cfg

        self.sentences_per_cap: Dict[str, List[str]] = {c: [] for c in taxonomy}
        self.sentinel_ids: List[str] = []
        self.sentinel_cap_map: Dict[str, str] = {}

    # ---------------------------------------------------------------------------
    def refresh(self):
        sample_size = self.pool_size * 10
        candidates: Dict[str, List[str]] = defaultdict(list)
        stream = load_dataset("c4", split="train", streaming=True, cache_dir=".cache")
        it = iter(stream.shuffle(buffer_size=10_000, seed=random.randint(0, 1_000_000)))
        while any(len(v) < sample_size for v in candidates.values()):
            try:
                ex = next(it)
            except StopIteration:
                break
            txt = ex["text"].replace("\n", " ").strip()
            matched = False
            for cap in self.taxonomy:
                if any(re.search(pat, txt, flags=re.IGNORECASE) for pat in _CAP_PATTERNS.get(cap, [])):
                    candidates[cap].append(txt)
                    matched = True
                    break
            if not matched and "global" in self.taxonomy:
                candidates["global"].append(txt)

        scored: Dict[str, List[Tuple[str, float, np.ndarray]]] = {c: [] for c in self.taxonomy}
        for cap in self.taxonomy:
            subset = random.sample(candidates.get(cap, []), min(sample_size, len(candidates.get(cap, []))))
            for txt in subset:
                ids = self.tokenizer(txt, truncation=True, max_length=64).input_ids
                score = len(ids)
                scored[cap].append((txt, score, np.asarray(ids[:32])))

        for cap in self.taxonomy:
            items = scored[cap]
            if not items:
                continue
            vecs = np.stack([it[2] for it in items])
            k = min(self.pool_size, len(items))
            km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(vecs) if k >= 2 else None
            chosen = []
            for cidx in range(k):
                idxs = np.where(km.labels_ == cidx)[0] if km else [cidx]
                best = max(idxs, key=lambda i: items[i][1])
                chosen.append(items[best][0])
            self.sentences_per_cap[cap] = chosen[: self.pool_size]

        self.sentinel_ids.clear(); self.sentinel_cap_map.clear()
        for cap in self.taxonomy:
            for idx, sent in enumerate(self.sentences_per_cap[cap]):
                sid = f"{cap}_{idx}"
                self.sentinel_ids.append(sid); self.sentinel_cap_map[sid] = cap

    # ---------------------------------------------------------------------------
    def sample_all(self) -> Tuple[List[str], List[str]]:
        prompts, caps = [], []
        for cap in self.taxonomy:
            for sent in self.sentences_per_cap[cap]:
                prompts.append(sent); caps.append(cap)
        return prompts, caps