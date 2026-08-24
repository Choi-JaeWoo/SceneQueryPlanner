"""Collect LLM self-generated trajectories over the training set.

Run from the repository root, e.g.:
    python -m core.run.collect_llm --config-name=config_wah task_planner=reactstrq ...
    python -m core.run.collect_llm --config-name=config_procthor task_planner=reactstrq ...
"""
import json
import logging
import os
import random

import hydra
from omegaconf import OmegaConf

from core.run.factory import build_env, build_llm_agent, build_planner

log = logging.getLogger(__name__)

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../conf')


@hydra.main(version_base=None, config_path=CONF_DIR, config_name='config_procthor')
def main(cfg):
    log.info(OmegaConf.to_yaml(cfg))

    env = build_env(cfg)
    llm_agent = build_llm_agent(cfg)
    tp = build_planner(cfg, env, llm_agent)

    if cfg.dataset_type == 'wah':
        with open(cfg.dataset.wah_trainset, 'r') as json_file:
            train_set = json.load(json_file)
    elif cfg.dataset_type == 'procthor':
        random.seed(cfg.procthor.random_seed_for_eval_subset)
        with open(cfg.procthor.eval_set, 'r') as json_file:
            train_set = json.load(json_file)
        # Sample one instruction paraphrase per task.
        for task_d in train_set:
            if isinstance(task_d.get("instruction"), list):
                task_d["instruction"] = [random.choice(task_d["instruction"])]
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")

    collect_dir = cfg.dataset.collect_dir
    os.makedirs(collect_dir, exist_ok=True)

    for task_d in train_set:
        tp.collect_llm(task_d, collect_dir)


if __name__ == "__main__":
    main()
