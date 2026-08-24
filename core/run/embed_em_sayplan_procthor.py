import os
import sys
import hydra
import json
import logging
import time
import shutil
import re
import pickle


from core.env.procthor_env import ProcThorEnv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from core.utils.procthor_utils import check_goal_condition


def parse_filename(file_name):
    match = re.match(r"traj_(\d+)_([^_]+)", file_name)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None

def load_taskset(eval_path):
    with open(eval_path, 'r') as f:
        return json.load(f)

def extract_all_plans(text: str):
    """
    Find the last plan: [ ... ] block in the text and return it as a list.
    Items are stripped; returns the comma-separated command strings as a list.
    """
    # Extract all plan: [ ... ] sections
    pattern = r'plan:\s*\[(.*?)\]'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        return []

    # Select the last plan block
    last_plan_raw = matches[-1]

    # Split by commas and strip surrounding whitespace
    plan_steps = [step.strip() for step in last_plan_raw.split(',') if step.strip()]
    return plan_steps

def parse_task_goal(text: str) -> str:
    """
    Return the text from 'Your task is to:' up to (but not including) '3D Scene Graph', preserving newlines.
    """
    start_token = "Your task is to:"
    end_token = "3D Scene Graph"

    start_idx = text.find(start_token)
    end_idx = text.find(end_token)

    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return ""

    # Return the text from start_token up to end_token as-is
    task_goal = text[start_idx:end_idx]
    return task_goal.strip()

def embed_traj(cfg):
    em_dir = cfg.llm_agent.em_dir
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')

    semantic_base_dir = os.path.join(em_dir, 'semantic_search')
    iterative_base_dir = os.path.join(em_dir, 'iterative_replanning')

    ##### Semantic Search ######
    semantic_traj_dir = os.path.join(semantic_base_dir, 'text_traj')
    semantic_embed_dir = os.path.join(semantic_base_dir, 'embed')
    os.makedirs(semantic_embed_dir, exist_ok=True)
    count = 0
    for file_name in os.listdir(semantic_traj_dir):
        file_path = os.path.join(semantic_traj_dir, file_name)
        
        with open(file_path, 'r') as file:
            text_traj = file.read()
        
        parsing_result = parse_task_goal(text_traj)

        task_goal_text = parsing_result
        goal_embedding = sbert.encode(task_goal_text.split('Your task is to: ')[1])
        
        tokens = tokenizer(text_traj)['input_ids']
        token_count = len(tokens)
        
        embed_name = file_name.replace('.txt', '.pkl')
        embedding = {'text_trajectory': text_traj,
            'embedding': goal_embedding,
            'text_traj_path': file_path,
            'token_count': token_count,}
        semantic_em_embed_path = os.path.join(semantic_embed_dir, embed_name)
        with open(semantic_em_embed_path, 'wb') as pickle_file:
            pickle.dump(embedding, pickle_file)
        count += 1
    print(f"✅ Total embeddings created in semantic search: {count}")

    ##### Iterative Replanning ######
    iterative_traj_dir = os.path.join(iterative_base_dir, 'text_traj')
    iterative_embed_dir = os.path.join(iterative_base_dir, 'embed')
    os.makedirs(iterative_embed_dir, exist_ok=True)
    count = 0
    for file_name in os.listdir(iterative_traj_dir):
        file_path = os.path.join(iterative_traj_dir, file_name)
        
        with open(file_path, 'r') as file:
            text_traj = file.read()
        
        parsing_result = parse_task_goal(text_traj)

        task_goal_text = parsing_result
        goal_embedding = sbert.encode(task_goal_text.split('Your task is to: ')[1])
        
        tokens = tokenizer(text_traj)['input_ids']
        token_count = len(tokens)
        
        embed_name = file_name.replace('.txt', '.pkl')
        embedding = {'text_trajectory': text_traj,
            'embedding': goal_embedding,
            'text_traj_path': file_path,
            'token_count': token_count,}
        iterative_em_embed_path = os.path.join(iterative_embed_dir, embed_name)
        with open(iterative_em_embed_path, 'wb') as pickle_file:
            pickle.dump(embedding, pickle_file)
        count += 1
    print(f"✅ Total embeddings created in iterative replanning: {count}")

def extract_success_traj(cfg, task_list):
    env = ProcThorEnv(cfg)
    
    collect_base_dir = cfg.dataset.collect_dir
    collect_sem_dir = os.path.join(collect_base_dir, 'semantic_search')
    collect_iter_dir = os.path.join(collect_base_dir, 'iterative_replanning')

    em_base_dir = cfg.llm_agent.em_dir
    em_sem_dir = os.path.join(em_base_dir, 'semantic_search')
    em_iter_dir = os.path.join(em_base_dir, 'iterative_replanning')
    success_sem_dir = os.path.join(em_sem_dir, 'text_traj')
    success_iter_dir = os.path.join(em_iter_dir, 'text_traj')

    iterative_file_names = os.listdir(collect_iter_dir)
    
    os.makedirs(success_sem_dir, exist_ok=True)
    os.makedirs(success_iter_dir, exist_ok=True)
        
    for iterative_file_name in iterative_file_names:
        file_path = os.path.join(collect_iter_dir, iterative_file_name)
        #############
        env_id, task_type = parse_filename(iterative_file_name)
        if env_id is None:
            continue
        task_d = next(
            (t for t in task_list if t.get("env_id") == env_id and t.get("mode") == task_type),
            None
        )
        if task_d is None:
            logging.warning(f"[SKIP] env_id {env_id} with task_type {task_type} not found")
            continue

        print(f"[INFO] env_id: {env_id} / task_type: {task_type}")
        print(f"[INFO] instruction: {task_d['instruction']}")
        
        with open(file_path, 'r') as file:
            text_traj = file.read()

        test_action_seq = extract_all_plans(text_traj)
        env.reset(task_d)

        for nl_action in test_action_seq:
            if nl_action in ['done', 'failure']:
                pass
            else:
                try:
                    env.step(nl_action)
                except Exception as e:
                    continue

        ssr = check_goal_condition(
            task_d, env.controller, env.init_event,
            env.cleaned_objects, env.cooled_objects,
            env.heated_objects, env.filled_coffee_objects
        )
        print(f"[INFO] Subgoal Success Rate (SSR): {ssr}")
        if ssr == 1:
            semantic_file_name = iterative_file_name.replace('iterative_replanning', 'semantic_search')

            iterative_file_path = os.path.join(collect_iter_dir, iterative_file_name)
            semantic_file_path = os.path.join(collect_sem_dir, semantic_file_name)
            dest_iter_path = os.path.join(success_iter_dir, iterative_file_name)
            dest_sem_path = os.path.join(success_sem_dir, semantic_file_name)

            print("Copying iterative replanning trajectories")
            if not os.path.exists(dest_iter_path):
                shutil.copy(iterative_file_path, dest_iter_path)
            else:
                print('File already exists.')
            print("Copying semantic search trajectories")
            if not os.path.exists(dest_sem_path):
                shutil.copy(semantic_file_path, dest_sem_path)
            else:
                print('File already exists.')
    
@hydra.main(version_base=None, config_path='../conf', config_name='config_procthor')
def main(cfg):
    eval_path = cfg.procthor.eval_set
    task_list = load_taskset(eval_path)
    extract_success_traj(cfg, task_list)
    embed_traj(cfg)
    
if __name__ == "__main__":
    main()
