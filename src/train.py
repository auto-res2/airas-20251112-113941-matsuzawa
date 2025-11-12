import copy
import math
import os
import random
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.model import load_model_and_tokenizer
from src.preprocess import SentinelPool, build_dataloaders

# ------------------------------------------------------------------------------------------------------------------
#  CONSTANTS
# ------------------------------------------------------------------------------------------------------------------
PRIMARY_METRIC = "exact_match_dev"  # consistent name across training & evaluation
BACKUP_QUEUE_LEN = 5               # #in-RAM snapshots for hard-safety rollback
DEFAULT_SENT_POOL = 64            # default pool size when sentinel block missing
DEFAULT_SENT_REFRESH = 200        # default refresh interval steps

# ------------------------------------------------------------------------------------------------------------------
#  SMALL HELPER UTILS
# ------------------------------------------------------------------------------------------------------------------

def _flatten_run_cfg(cfg: DictConfig) -> DictConfig:
    """Hoist fields from cfg.run.* to the root level so that the rest of the code
    can access them directly.  A *runs* hydra-group is aliased to *run* in the
    main config, therefore `cfg.run` always exists after composition."""
    if "run" in cfg and isinstance(cfg.run, DictConfig):
        for k, v in cfg.run.items():
            if k not in cfg:  # do not overwrite explicit CLI overrides
                cfg[k] = v
    cfg.run_id = cfg.get("run_id", cfg.run.run_id)
    return cfg


def _set_nested(cfg: DictConfig, dotted_key: str, value: Any):
    node = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in node or node[p] is None:
            node[p] = OmegaConf.create()
        node = node[p]
    node[parts[-1]] = value

# ------------------------------------------------------------------------------------------------------------------
#  KL-DIV  (token-wise; averaged over seq-length)
# ------------------------------------------------------------------------------------------------------------------

def _kl_divergence(curr_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        curr_logp = torch.log_softmax(curr_logits.float(), dim=-1)
        base_p = torch.softmax(base_logits.float(), dim=-1)
        kl_tok = (base_p * (torch.log(base_p + 1e-8) - curr_logp)).sum(-1)
        return kl_tok.mean(-1)  # [batch]

# ------------------------------------------------------------------------------------------------------------------
#  HiFi-SACO DUAL-PRICE CONTROLLER (only when cfg.method == "proposed")
# ------------------------------------------------------------------------------------------------------------------

class HiFiSACOController:
    def __init__(self, cfg: DictConfig, capabilities: List[str]):
        self.capabilities = capabilities
        self.gamma: float = cfg.training.hiFiSACO.gamma
        self.eta: float = cfg.training.hiFiSACO.eta
        self.eps: float = cfg.training.hiFiSACO.epsilon

        N = cfg.training.get("num_samples", 1e6)
        d = cfg.model.peft.rank
        self.B_target: float = math.pi / math.sqrt(N * d)

        self.b_c: Dict[str, float] = {c: self.B_target / len(capabilities) for c in capabilities}
        self.lam_global: float = 1.0
        self.lam_c: Dict[str, float] = {c: 1.0 for c in capabilities}

        self.alpha_ema = 0.1
        self.prev_loss: float | None = None
        self.prev_kl_per_sid: Dict[str, float] = {}
        self.ema_util = 0.0
        self.ema_util_c: Dict[str, float] = {c: 0.0 for c in capabilities}

    def update(self, loss: float, sentinel_ids: List[str], sent_cap_map: Dict[str, str], kls_after: torch.Tensor):
        if self.prev_loss is None:
            self.prev_loss = float(loss)
            for sid, k in zip(sentinel_ids, kls_after):
                self.prev_kl_per_sid[sid] = k.item()
            return

        delta_k_total = 0.0
        delta_k_per_cap = defaultdict(float)
        for sid, new_kl in zip(sentinel_ids, kls_after):
            old = self.prev_kl_per_sid.get(sid, 0.0)
            d_k = new_kl.item() - old
            delta_k_total += d_k
            cap = sent_cap_map[sid]
            delta_k_per_cap[cap] += d_k
            self.prev_kl_per_sid[sid] = new_kl.item()

        util = (self.prev_loss - loss) / max(delta_k_total, 1e-12)
        self.ema_util = (1 - self.alpha_ema) * self.ema_util + self.alpha_ema * util
        self.lam_global *= math.exp(self.eta * (util - self.ema_util))

        for c in self.capabilities:
            d_k_c = delta_k_per_cap[c]
            util_c = (self.prev_loss - loss) / max(d_k_c, 1e-12)
            self.ema_util_c[c] = (1 - self.alpha_ema) * self.ema_util_c[c] + self.alpha_ema * util_c
            self.lam_c[c] *= math.exp(self.eta * (util_c - self.ema_util_c[c]))
            self.b_c[c] = max(1e-6, self.b_c[c] - max(d_k_c, 0.0))

        self.prev_loss = float(loss)

    def lr_scale_factor(self, cap: str) -> float:
        alpha_t = self.gamma / (self.lam_global + self.eps)
        delta_c = 1.0 / (self.lam_c[cap] + self.eps)
        return min(alpha_t, delta_c)

    def is_safe(self, sentinel_ids: List[str], sent_cap_map: Dict[str, str], kls: torch.Tensor) -> bool:
        for sid, k in zip(sentinel_ids, kls):
            cap = sent_cap_map[sid]
            if k.item() > 0.5 * self.b_c[cap]:
                return False
        return True

# ------------------------------------------------------------------------------------------------------------------
#  DEV-SET (GSM-8K) EVALUATION
# ------------------------------------------------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    return text.split("####")[-1].strip().split("\n")[-1].strip()


def evaluate_dev_set(model: nn.Module, tokenizer, loader, device) -> Tuple[float, List[str], List[str], List[str]]:
    model.eval()
    total, correct = 0, 0
    prompts_out, preds_out, golds_out = [], [], []
    with torch.no_grad():
        for batch in loader:
            enc = tokenizer(batch["prompt_text"], return_tensors="pt", padding=True, truncation=True,
                            max_length=tokenizer.model_max_length).to(device)
            gen = model.generate(**enc, max_new_tokens=64)
            for i, prompt_txt in enumerate(batch["prompt_text"]):
                gen_txt = tokenizer.decode(gen[i][enc.input_ids.shape[1]:], skip_special_tokens=True)
                pred = _extract_answer(gen_txt)
                gold = _extract_answer(batch["answer_text"][i])
                prompts_out.append(prompt_txt)
                preds_out.append(pred)
                golds_out.append(gold)
                correct += int(pred == gold)
                total += 1
    model.train()
    return correct / max(total, 1), prompts_out, preds_out, golds_out

# ------------------------------------------------------------------------------------------------------------------
#  OPTIMISER BUILD
# ------------------------------------------------------------------------------------------------------------------

def build_optimizer(cfg: DictConfig, model: nn.Module, param_cap_map: Dict[str, str]):
    param_groups = []
    for cap in set(param_cap_map.values()):
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if param_cap_map[name] != cap:
                continue
            (decay if p.ndim >= 2 and not name.endswith("bias") else no_decay).append(p)
        if decay:
            param_groups.append({"params": decay, "weight_decay": cfg.training.weight_decay, "cap": cap})
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0, "cap": cap})

    optimiser = AdamW(param_groups,
                      lr=cfg.training.base_learning_rate,
                      betas=(cfg.training.beta1, cfg.training.beta2),
                      eps=cfg.training.epsilon)
    scheduler = CosineAnnealingLR(optimiser, T_max=1) if cfg.training.lr_scheduler == "cosine" else LambdaLR(optimiser, lambda _: 1.0)
    return optimiser, scheduler

# ------------------------------------------------------------------------------------------------------------------
#  PARAM->CAPABILITY MAPPING  (cheap gradient attribution)
# ------------------------------------------------------------------------------------------------------------------

def infer_param_capabilities(model: nn.Module, pool: SentinelPool, tokenizer, device, samples_per_cap: int = 8) -> Dict[str, str]:
    if len(pool.taxonomy) == 1:
        return {n: pool.taxonomy[0] for n, p in model.named_parameters() if p.requires_grad}

    model.eval()
    scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for cap in pool.taxonomy:
        if not pool.sentences_per_cap[cap]:
            continue
        prompts = random.sample(pool.sentences_per_cap[cap], min(samples_per_cap, len(pool.sentences_per_cap[cap])))
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        loss = model(**enc, labels=enc.input_ids).loss
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                scores[n][cap] += float(p.grad.pow(2).mean().detach().cpu())
        model.zero_grad(set_to_none=True)

    mapping: Dict[str, str] = {}
    for name, cap_scores in scores.items():
        mapping[name] = max(cap_scores.items(), key=lambda kv: kv[1])[0]
    rest_caps = list(pool.taxonomy)
    for i, (n, p) in enumerate(model.named_parameters()):
        if p.requires_grad and n not in mapping:
            mapping[n] = rest_caps[i % len(rest_caps)]
    model.train()
    return mapping

# ------------------------------------------------------------------------------------------------------------------
#  TRAINING LOGIC FOR A *SINGLE* CONFIGURATION.  RETURNS PRIMARY METRIC VALUE
# ------------------------------------------------------------------------------------------------------------------

def run_training(cfg: DictConfig) -> float:
    # reproducibility -----------------------------------------------------------
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # WandB ---------------------------------------------------------------------
    wb_run = None
    if cfg.wandb.mode != "disabled":
        wb_run = wandb.init(entity=cfg.wandb.entity,
                            project=cfg.wandb.project,
                            id=cfg.run_id,
                            resume="allow",
                            mode=cfg.wandb.mode,
                            config=OmegaConf.to_container(cfg, resolve=True))

    # data ----------------------------------------------------------------------
    train_loader, dev_loader, tokenizer = build_dataloaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model ---------------------------------------------------------------------
    model, base_model, tokenizer = load_model_and_tokenizer(cfg, tokenizer, device)
    base_model.eval().requires_grad_(False)

    # sentinel pool -------------------------------------------------------------
    taxonomy = list(cfg.model.get("capability_taxonomy", ["global"]))
    sentinel_cfg = cfg.dataset.get("sentinel", OmegaConf.create({}))
    pool_size = sentinel_cfg.get("pool_size_per_capability", DEFAULT_SENT_POOL)
    refresh_int = sentinel_cfg.get("refresh_interval_steps", DEFAULT_SENT_REFRESH)
    pool = SentinelPool(tokenizer, taxonomy, pool_size, refresh_int,
                        cfg.dataset.get("streaming_source", {}), device)
    pool.refresh()

    # optimiser -----------------------------------------------------------------
    param_cap_map = infer_param_capabilities(model, pool, tokenizer, device)
    optimiser, scheduler = build_optimizer(cfg, model, param_cap_map)
    total_updates = cfg.training.epochs * math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    if isinstance(scheduler, CosineAnnealingLR):
        scheduler.T_max = total_updates

    controller = HiFiSACOController(cfg, taxonomy) if cfg.method == "proposed" else None
    rollback_buffer: deque[Tuple[Dict[str, torch.Tensor], Any]] = deque(maxlen=BACKUP_QUEUE_LEN)
    grad_acc = cfg.training.gradient_accumulation_steps

    def _wb_log(d: Dict[str, Any], step: int):
        if wb_run is not None:
            wandb.log(d, step=step)

    global_step, best_em = 0, 0.0

    for epoch in range(cfg.training.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if cfg.training.get("limit_batches") and batch_idx >= cfg.training.limit_batches:
                break  # trial-mode shortcut

            global_step += 1
            model.train()
            inp = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=inp, attention_mask=attn, labels=labels).loss
            (loss / grad_acc).backward()

            if global_step % grad_acc == 0:
                prompts, caps = pool.sample_all()
                enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                kl_before = _kl_divergence(model(**enc).logits, base_model(**enc).logits)
                sid_cap_map = {sid: c for sid, c in zip(pool.sentinel_ids, caps)}

                if controller is not None:
                    base_lr = cfg.training.base_learning_rate
                    for g in optimiser.param_groups:
                        cap = g.get("cap", random.choice(taxonomy))
                        g["lr"] = base_lr * controller.lr_scale_factor(cap)

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
                optimiser.step(); optimiser.zero_grad(set_to_none=True); scheduler.step()

                kl_after = _kl_divergence(model(**enc).logits, base_model(**enc).logits)

                safe = True
                if controller is not None:
                    controller.update(loss.item(), pool.sentinel_ids, sid_cap_map, kl_after)
                    safe = controller.is_safe(pool.sentinel_ids, sid_cap_map, kl_after)
                else:  # KARMA
                    ma_tau = cfg.training.karma.moving_average_tau
                    if not hasattr(run_training, "_karma_ma"):
                        run_training._karma_ma = kl_after.mean().item()
                    else:
                        run_training._karma_ma = ((1 - 1/ma_tau) * run_training._karma_ma +
                                                  (1/ma_tau) * kl_after.mean().item())
                    safe = run_training._karma_ma <= cfg.training.karma.global_kl_budget_B

                if not safe and cfg.training.get("hard_safety", True):
                    if rollback_buffer:
                        state, opt_state = rollback_buffer.pop()
                        model.load_state_dict(state, strict=False)
                        optimiser.load_state_dict(opt_state)
                        print(f"[SAFETY-ROLLBACK] step={global_step}")
                        continue
                    else:
                        print("[SAFETY] No checkpoint – aborting run.")
                        if wb_run:
                            wb_run.finish()
                        return best_em
                else:
                    rollback_buffer.append((copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                                             copy.deepcopy(optimiser.state_dict())))

                _wb_log({
                    "train_loss": loss.item(),
                    "sentinel_avg_kl": kl_after.mean().item(),
                    "sentinel_max_kl_ratio": max([
                        kl_after[i].item() /
                        (controller.b_c[sid_cap_map[pool.sentinel_ids[i]]] if controller else cfg.training.karma.global_kl_budget_B)
                        for i in range(len(kl_after))
                    ]),
                }, step=global_step)

                if global_step % pool.refresh_interval == 0:
                    pool.refresh()

        em, ppts, preds, golds = evaluate_dev_set(model, tokenizer, dev_loader, device)
        best_em = max(best_em, em)
        _wb_log({PRIMARY_METRIC: em}, step=global_step)

        if wb_run:
            tbl = wandb.Table(columns=["prompt", "gold", "pred", "correct"])
            for pr, g, pd in zip(ppts, golds, preds):
                tbl.add_data(pr, g, pd, int(pd == g))
            wb_run.log({"dev_predictions": tbl}, step=global_step)
            wb_run.summary.update({"epoch": epoch, PRIMARY_METRIC: em, f"best_{PRIMARY_METRIC}": best_em})

    out_dir = os.path.join(cfg.results_dir, cfg.run_id, "peft")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)

    if wb_run:
        print("WandB URL:", wb_run.url)
        wb_run.finish()

    return best_em

# ------------------------------------------------------------------------------------------------------------------
#  OPTUNA SUPPORT
# ------------------------------------------------------------------------------------------------------------------

def _sample_trial_params(trial: optuna.Trial, space: DictConfig) -> Dict[str, Any]:
    picked = {}
    for name, spec in space.items():
        t = spec.type.lower()
        if t in {"loguniform", "log_uniform"}:
            picked[name] = trial.suggest_float(name, float(spec.low), float(spec.high), log=True)
        elif t in {"uniform", "float"}:
            picked[name] = trial.suggest_float(name, float(spec.low), float(spec.high), log=False)
        elif t in {"int", "integer"}:
            picked[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
        elif t == "categorical":
            picked[name] = trial.suggest_categorical(name, list(spec.choices))
        else:
            raise ValueError(f"Unsupported Optuna type '{t}' for '{name}'.")
    return picked

# ------------------------------------------------------------------------------------------------------------------
#  HYDRA ENTRYPOINT (config_path="../config")
# ------------------------------------------------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig):
    cfg = _flatten_run_cfg(cfg)

    # mode-specific overrides ----------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        _set_nested(cfg, "training.limit_batches", 2)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # hyper-parameter search -----------------------------------------------------
    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0:
        metric_name = cfg.optuna.metric

        def objective(trial: optuna.Trial):
            trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            trial_cfg.wandb.mode = "disabled"  # never log individual trials
            trial_cfg.run_id = f"{cfg.run_id}_optuna_{trial.number}"
            for k, v in _sample_trial_params(trial, cfg.optuna.search_space).items():
                _set_nested(trial_cfg, k, v)
            _flatten_run_cfg(trial_cfg)
            metric_value = run_training(trial_cfg)
            return metric_value

        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True)
        for k, v in study.best_params.items():
            _set_nested(cfg, k, v)
        print("[Optuna] best value:", study.best_value, "params:", study.best_params)

    # final training run ---------------------------------------------------------
    run_training(cfg)


if __name__ == "__main__":
    _main()