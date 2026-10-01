"""Evaluate a task planner on WAH-NL (VirtualHome) or AttPlan-Bench (ProcTHOR).

Run from the repository root, e.g.:
    python -m core.run.eval --config-name=config_wah task_planner=reactstrq ...
    python -m core.run.eval --config-name=config_procthor task_planner=reactstrq ...
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import itertools
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


def summarize_procthor(results):
    num_task = len(results)
    avg_goal_success_rate = sum(r['goal_success_rate'] for r in results) / num_task
    avg_subgoal_success_rate = sum(r['subgoal_success_rate'] for r in results) / num_task
    successful_results = [r for r in results if r['goal_success_rate'] == 1]
    if successful_results:
        avg_decision_step = sum(r['decision_step'] for r in successful_results) / len(successful_results)
    else:
        avg_decision_step = 0
    return num_task, avg_goal_success_rate, avg_subgoal_success_rate, avg_decision_step


def eval_procthor(cfg):
    from ai2thor.exceptions import RestartError
    from core.utils.procthor_utils import check_goal_condition, task_file_stem
    from core.utils.vis_procthor import save_images_as_mp4

    random.seed(cfg.procthor.random_seed_for_eval_subset)

    llm_agent = build_llm_agent(cfg)
    em_dir_by_mode = cfg.llm_agent.get('em_dir_by_mode') or {}
    exclude_modes = cfg.procthor.get('exclude_modes') or []

    total_results = []
    all_results = []
    overall_start = time.time()
    for eval_path in cfg.procthor.eval_set:
        with open(eval_path, 'r') as json_file:
            test_set = json.load(json_file)
        test_set = [task_d for task_d in test_set if task_d['mode'] not in exclude_modes]

        for mode, block in itertools.groupby(test_set, key=lambda task_d: task_d['mode']):
            block = list(block)
            block_name = f"{eval_path} [{mode}]"
            em_dir = em_dir_by_mode.get(mode, cfg.llm_agent.em_dir)
            llm_agent.em_dir = em_dir
            log.info(f"\n\nEvaluating: {block_name} (examples: {em_dir})\n{'-' * 60}")

            # Sample one instruction paraphrase per task.
            for task_d in block:
                if isinstance(task_d.get("instruction"), list):
                    task_d["instruction"] = [random.choice(task_d["instruction"])]

            env = build_env(cfg)
            tp = build_planner(cfg, env, llm_agent)

            eval_start = time.time()
            results = []
            for task_d in block:
                simulator_error = False
                try:
                    tp.run(task_d, log)
                    ssr = check_goal_condition(task_d, env.controller, env.init_event,
                                               env.cleaned_objects, env.cooled_objects,
                                               env.heated_objects, env.filled_coffee_objects,
                                               env.init_state_hard)
                except (TimeoutError, RestartError) as e:
                    log.info(f"Simulator error, the task is counted as failed and the simulator is restarted: {e}")
                    simulator_error = True
                    ssr = 0.0
                sr = 1 if ssr == 1.0 else 0
                result = {'env_id': task_d['env_id'],
                          'mode': task_d['mode'],
                          'nl_inst': task_d['instruction'][0],
                          'goal_success_rate': sr,
                          'subgoal_success_rate': ssr,
                          'decision_step': tp.cur_decision_step}
                if task_d.get('init_action'):
                    result['init_action'] = task_d['init_action']
                if simulator_error:
                    result['simulator_error'] = True
                log.info(result)
                results.append(result)
                if simulator_error:
                    try:
                        env.controller.stop()
                    except Exception as e:
                        log.info(f"Failed to stop the simulator cleanly: {e}")
                    env = build_env(cfg)
                    tp = build_planner(cfg, env, llm_agent)
                elif cfg.procthor.vis_log:
                    img_list = [Image.fromarray(step['images'].astype(np.uint8)) for step in env.vis_log]
                    text_list = [step['action'] for step in env.vis_log]
                    right_text_list = [step['observation'] for step in env.vis_log]
                    out_path = os.path.join(HydraConfig.get().run.dir,
                                            f"{task_file_stem(task_d)}.mp4")
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

            all_results.extend(results)
            num_task, avg_goal_success_rate, avg_subgoal_success_rate, avg_decision_step = summarize_procthor(results)
            num_simulator_error = sum(r.get('simulator_error', False) for r in results)

            log.info(f"\n### Results for {block_name}")
            log.info(f"  - average goal success rate: {avg_goal_success_rate * 100:.2f} %")
            log.info(f"  - average subgoal success rate: {avg_subgoal_success_rate * 100:.2f} %")
            log.info(f"  - average decision step: {avg_decision_step:.2f}")
            log.info(f"  - simulator errors (counted as failed): {num_simulator_error}")
            log.info(f"  - time taken: {eval_time / 60:.1f} mins")

            total_results.append({
                "eval_set": block_name,
                "num_task": num_task,
                "goal_success_rate": avg_goal_success_rate,
                "subgoal_success_rate": avg_subgoal_success_rate,
                "decision_step": avg_decision_step,
                "simulator_errors": num_simulator_error,
                "time_minutes": eval_time / 60,
            })

    log.info("\n----------- Summary of all evaluations -----------")
    for result in total_results:
        log.info(f"- {result['eval_set']}")
        log.info(f"  - Tasks: {result['num_task']}")
        log.info(f"  - Goal Success: {result['goal_success_rate'] * 100:.2f} %")
        log.info(f"  - Subgoal Success: {result['subgoal_success_rate'] * 100:.2f} %")
        log.info(f"  - Decision Step: {result['decision_step']:.2f}")
        log.info(f"  - Simulator Errors: {result['simulator_errors']}")
        log.info(f"  - Time: {result['time_minutes']:.1f} mins\n")

    num_task, avg_goal_success_rate, avg_subgoal_success_rate, avg_decision_step = summarize_procthor(all_results)
    log.info("- Overall")
    log.info(f"  - Tasks: {num_task}")
    log.info(f"  - Goal Success: {avg_goal_success_rate * 100:.2f} %")
    log.info(f"  - Subgoal Success: {avg_subgoal_success_rate * 100:.2f} %")
    log.info(f"  - Decision Step: {avg_decision_step:.2f}")
    log.info(f"  - Simulator Errors: {sum(r.get('simulator_error', False) for r in all_results)}\n")

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
