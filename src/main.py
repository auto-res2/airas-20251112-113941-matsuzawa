import os
import subprocess
import sys
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def _launcher(cfg: DictConfig):
    """Launches **one** training run (src.train) in a subprocess."""
    run_id = cfg.run.run_id

    overrides: List[str] = [
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    if cfg.mode == "trial":
        overrides += [
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.epochs=1",
            "training.limit_batches=2",
        ]
    elif cfg.mode == "full":
        overrides.append("wandb.mode=online")
    else:
        raise ValueError("mode must be trial|full")

    abs_res = os.path.abspath(cfg.results_dir); os.makedirs(abs_res, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(abs_res, f"{run_id}_launcher_cfg.yaml"))

    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    print("[LAUNCH]", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    _launcher()