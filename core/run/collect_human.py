"""Interactively collect human demonstration trajectories.

Run from the repository root, e.g.:
    python -m core.run.collect_human --config-name=config_wah task_planner=reactstrq ...
    python -m core.run.collect_human --config-name=config_procthor task_planner=reactstrq ...
"""
import json
import logging
import os

import hydra
from omegaconf import OmegaConf

from core.run.factory import build_env, build_planner

log = logging.getLogger(__name__)

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../conf')


def collect_wah(cfg, tp):
    with open(cfg.dataset.wah_trainset, 'r') as json_file:
        train_set = json.load(json_file)

    collect_dir = cfg.dataset.collect_dir
    os.makedirs(collect_dir, exist_ok=True)

    while True:
        task_id = int(input('Type target task ID (0~249): '))
        task_d = train_set[task_id]
        tp.collect_human(task_d, collect_dir)


def collect_procthor(cfg, tp):
    with open(cfg.procthor.eval_set, 'r') as json_file:
        train_set = json.load(json_file)

    collect_dir = cfg.dataset.collect_dir
    os.makedirs(collect_dir, exist_ok=True)

    while True:
        task_mode = input("Type task mode (e.g., PickAndPlaceSingleTask): ").strip()
        task_id = int(input('Type target env id (0~999): '))
        task_d = next((task for task in train_set
                       if task['mode'] == task_mode and task['env_id'] == task_id), None)
        if task_d is None:
            print(f"[WARNING] No task found with mode '{task_mode}' and env_id {task_id}")
            continue
        tp.collect_human(task_d, collect_dir)


@hydra.main(version_base=None, config_path=CONF_DIR, config_name='config_procthor')
def main(cfg):
    log.info(OmegaConf.to_yaml(cfg))
    env = build_env(cfg)
    tp = build_planner(cfg, env, llm_agent=None)
    if cfg.dataset_type == 'wah':
        collect_wah(cfg, tp)
    elif cfg.dataset_type == 'procthor':
        collect_procthor(cfg, tp)
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")


if __name__ == "__main__":
    main()
