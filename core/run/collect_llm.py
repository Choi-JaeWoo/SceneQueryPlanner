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


def snapshot(collect_dir):
    return {os.path.join(dir_path, file_name): os.stat(os.path.join(dir_path, file_name)).st_mtime_ns
            for dir_path, _, file_names in os.walk(collect_dir) for file_name in file_names}


def is_task_file(cfg, task_d, file_path):
    if cfg.dataset_type == 'procthor':
        from core.utils.procthor_utils import task_file_stem
        stem = task_file_stem(task_d)
    else:
        stem = f"{task_d['task_id']:03d}"
    return os.path.basename(file_path).startswith((f"traj_{stem}.", f"traj_{stem}_", f"{stem}_"))


@hydra.main(version_base=None, config_path=CONF_DIR, config_name='config_procthor')
def main(cfg):
    log.info(OmegaConf.to_yaml(cfg))

    env = build_env(cfg)
    llm_agent = build_llm_agent(cfg)
    tp = build_planner(cfg, env, llm_agent)

    if cfg.dataset_type == 'wah':
        simulator_errors = ()
        with open(cfg.dataset.wah_trainset, 'r') as json_file:
            train_set = json.load(json_file)
    elif cfg.dataset_type == 'procthor':
        from ai2thor.exceptions import RestartError
        simulator_errors = (TimeoutError, RestartError)
        random.seed(cfg.procthor.random_seed_for_eval_subset)
        with open(cfg.procthor.eval_set, 'r') as json_file:
            train_set = json.load(json_file)
        exclude_modes = cfg.procthor.get('exclude_modes') or []
        train_set = [task_d for task_d in train_set if task_d['mode'] not in exclude_modes]
        # Sample one instruction paraphrase per task.
        for task_d in train_set:
            if isinstance(task_d.get("instruction"), list):
                task_d["instruction"] = [random.choice(task_d["instruction"])]
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")

    collect_dir = cfg.dataset.collect_dir
    os.makedirs(collect_dir, exist_ok=True)

    skipped = []
    for task_d in train_set:
        before = snapshot(collect_dir)
        try:
            tp.collect_llm(task_d, collect_dir)
        except simulator_errors as e:
            log.info(f"Simulator error, skipping this task and restarting the simulator: {e}")
            skipped.append({k: task_d[k] for k in ('env_id', 'mode', 'task_id') if k in task_d})
            for file_path, mtime in snapshot(collect_dir).items():
                if before.get(file_path) != mtime and is_task_file(cfg, task_d, file_path):
                    os.remove(file_path)
                    log.info(f"Removed partial trajectory: {file_path}")
            try:
                env.controller.stop()
            except Exception as stop_error:
                log.info(f"Failed to stop the simulator cleanly: {stop_error}")
            env = build_env(cfg)
            tp = build_planner(cfg, env, llm_agent)

    log.info(f"Skipped tasks due to simulator errors: {len(skipped)} {skipped}")


if __name__ == "__main__":
    main()
