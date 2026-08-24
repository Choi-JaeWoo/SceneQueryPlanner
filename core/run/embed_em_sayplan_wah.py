import os
import sys
import hydra
import json
import logging
import time
import shutil
import re
import pickle


from core.env.wah_env import WahEnv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from core.utils.wah_utils import check_goal_condition


def extract_all_plans(text: str):
    """
    텍스트에서 가장 마지막 plan: [ ... ] 블록을 찾아 리스트로 반환한다.
    항목은 strip되어 있으며, 쉼표로 구분된 명령어 문자열을 리스트로 반환.
    """
    # 모든 plan: [ ... ] 구간 추출
    pattern = r'plan:\s*\[(.*?)\]'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        return []

    # 마지막 plan 블록 선택
    last_plan_raw = matches[-1]

    # 쉼표로 분할하고 양끝 공백 제거
    plan_steps = [step.strip() for step in last_plan_raw.split(',') if step.strip()]
    return plan_steps


def parse_task_goal(text: str) -> str:
    """
    'Your task is to:'부터 '3D Scene Graph' 이전까지의 텍스트를 줄바꿈 포함 그대로 반환한다.
    """
    start_token = "Your task is to:"
    end_token = "3D Scene Graph"

    start_idx = text.find(start_token)
    end_idx = text.find(end_token)

    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return ""

    # start_token부터 end_token 이전까지 그대로 반환
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

    ##### Iterative Replanning ######
    iterative_traj_dir = os.path.join(iterative_base_dir, 'text_traj')
    iterative_embed_dir = os.path.join(iterative_base_dir, 'embed')
    os.makedirs(iterative_embed_dir, exist_ok=True)
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

def get_task_id_from_file(file_path):
    file_name = os.path.basename(file_path)
    task_id_str = file_name.split('_')[0]

    try:
        task_id = int(task_id_str)
        return task_id
    except ValueError:
        logging.error(f"Invalid task ID in file name: {file_name}")
        return None


def extract_success_traj(cfg):
    env = WahEnv(cfg)
    with open(cfg.dataset.wah_trainset, 'r') as json_file:
        train_set = json.load(json_file)
    
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
        
        with open(file_path, 'r') as file:
            text_traj = file.read()
        
        test_action_seq = extract_all_plans(text_traj)
        task_id = get_task_id_from_file(file_path)
        
        task_d = train_set[task_id]
        env.reset(task_d)

        for nl_action in test_action_seq:
            if nl_action in ['done', 'failure']:
                pass
            else:
                try:
                    env.step(nl_action)
                except Exception as e:
                    continue

        ssr = check_goal_condition(task_d['task_goal'], env.get_graph(), env.name_id_dict_sim2nl, env.name_id_dict_nl2sim)
        
        if ssr == 1:
            iterative_file_name
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
    
@hydra.main(version_base=None, config_path='../conf', config_name='config_wah')
def main(cfg):
    extract_success_traj(cfg)
    embed_traj(cfg)
    
if __name__ == "__main__":
    main()
