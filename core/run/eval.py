"""Evaluate a task planner on WAH-NL (VirtualHome) or AttPlan-Bench (ProcTHOR).

Run from the repository root, e.g.:
    python -m core.run.eval --config-name=config_wah task_planner=reactstrq ...
    python -m core.run.eval --config-name=config_procthor task_planner=reactstrq ...
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import json
import logging
import random
import time

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image

from core.run.factory import build_env, build_llm_agent, build_planner

log = logging.getLogger(__name__)

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../conf')


def eval_wah(cfg):
    from core.utils.wah_utils import check_goal_condition, save_vis_log

    env = build_env(cfg)
    llm_agent = build_llm_agent(cfg)
    tp = build_planner(cfg, env, llm_agent)

    with open(cfg.dataset.wah_testset, 'r') as json_file:
        test_set = json.load(json_file)

    results = []
    start = time.time()
    for task_d in test_set:
        tp.run(task_d, log)
        ssr = check_goal_condition(task_d['task_goal'], env.get_graph(),
                                   env.name_id_dict_sim2nl, env.name_id_dict_nl2sim)
        sr = 1 if ssr == 1.0 else 0
        result = {'task_id': task_d['task_id'],
                  'nl_inst': task_d['nl_instructions'][0],
                  'goal_success_rate': sr,
                  'subgoal_success_rate': ssr}
        log.info(result)
        results.append(result)
        if cfg.environment.vis_log:
            save_vis_log(HydraConfig.get().run.dir, env.vis_log,
                         task_d['task_id'], task_d['nl_instructions'][0])

    log.info(results)
    num_task = len(results)
    avg_goal_success_rate = sum(r['goal_success_rate'] for r in results) / num_task
    avg_subgoal_success_rate = sum(r['subgoal_success_rate'] for r in results) / num_task

    log.info(f'average goal success rate: {avg_goal_success_rate * 100:.2f} %')
    log.info(f'average subgoal success rate: {avg_subgoal_success_rate * 100:.2f} %')
    log.info(f'took {(time.time() - start) / 60:.1f} mins')


def eval_procthor(cfg):
    from core.utils.procthor_utils import check_goal_condition
    from core.utils.vis_procthor import save_images_as_mp4

    random.seed(cfg.procthor.random_seed_for_eval_subset)

    llm_agent = build_llm_agent(cfg)

    total_results = []
    overall_start = time.time()
    for eval_path in cfg.procthor.eval_set:
        log.info(f"\n\nEvaluating: {eval_path}\n{'-' * 60}")
        with open(eval_path, 'r') as json_file:
            test_set = json.load(json_file)

        # Sample one instruction paraphrase per task.
        for task_d in test_set:
            if isinstance(task_d.get("instruction"), list):
                task_d["instruction"] = [random.choice(task_d["instruction"])]

        env = build_env(cfg)
        tp = build_planner(cfg, env, llm_agent)

        eval_start = time.time()
        results = []
        for task_d in test_set:
            tp.run(task_d, log)
            ssr = check_goal_condition(task_d, env.controller, env.init_event,
                                       env.cleaned_objects, env.cooled_objects,
                                       env.heated_objects, env.filled_coffee_objects)
            sr = 1 if ssr == 1.0 else 0
            result = {'env_id': task_d['env_id'],
                      'mode': task_d['mode'],
                      'nl_inst': task_d['instruction'][0],
                      'goal_success_rate': sr,
                      'subgoal_success_rate': ssr,
                      'decision_step': tp.cur_decision_step}
            log.info(result)
            results.append(result)
            if cfg.procthor.vis_log:
                img_list = [Image.fromarray(step['images'].astype(np.uint8)) for step in env.vis_log]
                text_list = [step['action'] for step in env.vis_log]
                right_text_list = [step['observation'] for step in env.vis_log]
                out_path = os.path.join(HydraConfig.get().run.dir,
                                        f"{task_d['env_id']}_{task_d['mode']}.mp4")
                save_images_as_mp4(img_list=img_list,
                                   text_list=text_list,
                                   right_text_list=right_text_list,
                                   file_name=out_path,
                                   vis_type="dec_obs",
                                   action_font_size=14,
                                   right_text_font_size=15,
                                   max_right_lines=12,
                                   duration_ms=1500)

        eval_time = time.time() - eval_start

        env.controller.stop()
        del env
        del tp
        gc.collect()

        num_task = len(results)
        avg_goal_success_rate = sum(r['goal_success_rate'] for r in results) / num_task
        avg_subgoal_success_rate = sum(r['subgoal_success_rate'] for r in results) / num_task

        successful_results = [r for r in results if r['goal_success_rate'] == 1]
        if successful_results:
            avg_decision_step = sum(r['decision_step'] for r in successful_results) / len(successful_results)
        else:
            avg_decision_step = 0

        log.info(f"\n### Results for {eval_path}")
        log.info(f"  - average goal success rate: {avg_goal_success_rate * 100:.2f} %")
        log.info(f"  - average subgoal success rate: {avg_subgoal_success_rate * 100:.2f} %")
        log.info(f"  - average decision step: {avg_decision_step:.2f}")
        log.info(f"  - time taken: {eval_time / 60:.1f} mins")

        total_results.append({
            "eval_set": eval_path,
            "num_task": num_task,
            "goal_success_rate": avg_goal_success_rate,
            "subgoal_success_rate": avg_subgoal_success_rate,
            "decision_step": avg_decision_step,
            "time_minutes": eval_time / 60,
        })

    log.info("\n----------- Summary of all evaluations -----------")
    for result in total_results:
        log.info(f"- {result['eval_set']}")
        log.info(f"  - Tasks: {result['num_task']}")
        log.info(f"  - Goal Success: {result['goal_success_rate'] * 100:.2f} %")
        log.info(f"  - Subgoal Success: {result['subgoal_success_rate'] * 100:.2f} %")
        log.info(f"  - Decision Step: {result['decision_step']:.2f}")
        log.info(f"  - Time: {result['time_minutes']:.1f} mins\n")

    log.info(f"Total time: {(time.time() - overall_start) / 60:.1f} mins")


@hydra.main(version_base=None, config_path=CONF_DIR, config_name='config_procthor')
def main(cfg):
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.dataset_type == 'wah':
        eval_wah(cfg)
    elif cfg.dataset_type == 'procthor':
        eval_procthor(cfg)
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")


if __name__ == "__main__":
    main()
