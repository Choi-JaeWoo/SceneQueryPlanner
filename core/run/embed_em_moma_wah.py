import os
import sys
import hydra
import json
import logging
import time
import shutil
import re
import pickle
from collections import defaultdict
import numpy as np



from core.env.wah_env import WahEnv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from core.utils.wah_utils import check_goal_condition



def get_task_id_from_file(file_path):
    file_name = os.path.basename(file_path)
    task_id_str = file_name.split('_')[1].split('.')[0]

    try:
        task_id = int(task_id_str)
        return task_id
    except ValueError:
        logging.error(f"Invalid task ID in file name: {file_name}")
        return None

def extract_act_seq(file_path):
    act_seq = []
    with open(file_path, 'r') as file:
        txt_content = file.read()
        for line in txt_content.splitlines():
            if line.startswith('Act: '):
                act_seq.append(line.split('Act: ')[1])
    return act_seq


def extract_success_traj(cfg):
    env = WahEnv(cfg)
    with open(cfg.dataset.wah_trainset, 'r') as json_file:
        train_set = json.load(json_file)
    collect_traj_dir, em_dir = cfg.dataset.collect_dir, cfg.llm_agent.em_dir
    file_names = os.listdir(collect_traj_dir)
    success_traj_dir = os.path.join(em_dir, 'text_traj')
    os.makedirs(success_traj_dir, exist_ok=True)

    # ---- 여기부터 추가: task_id별로 step 정렬해서 나누기 ----
    pat = re.compile(r'^traj_(\d{3})_(\d{3})\.txt$')  # traj_{task_id}_{step_id}.txt
    grouped = defaultdict(list)  # {task_id: [(s

    for fn in file_names:
        m = pat.match(fn)
        if not m:
            # 패턴에 안 맞는 파일은 무시
            continue
        task_id = m.group(1)          # '061'
        step_id = int(m.group(2))     # 69
        grouped[task_id].append((step_id, fn))

    # step_id 기준으로 정렬 후 파일명만 뽑기
    files_by_task = {
        tid: [fn for (step, fn) in sorted(pairs, key=lambda x: x[0])]
        for tid, pairs in grouped.items()
    }

    # 2. files_by_task에 있는 task_id만 순회
    for task_id, fn_list in files_by_task.items():
        task_d = train_set[int(task_id)]

        # 액션 시퀀스 뽑기 (Act: 라인만)
        act_seq = []
        for fn in fn_list:
            fp = os.path.join(collect_traj_dir, fn)
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Act:"):
                        act_seq.append(line.replace("Act:", "").strip())
                        break   # 액션 하나만 있으니 break
        
        # 환경 초기화 및 실행
        env.reset(task_d)
        for nl_action in act_seq:
            if nl_action in ["done", "failure"]:
                continue
            env.step(nl_action)

        ssr = check_goal_condition(task_d['task_goal'], env.get_graph(), env.name_id_dict_sim2nl, env.name_id_dict_nl2sim)
        if ssr == 1:
            for fn in fn_list:
                src_path = os.path.join(collect_traj_dir, fn)
                dest_path = os.path.join(success_traj_dir, fn)
                if not os.path.exists(dest_path):
                    shutil.copy(src_path, dest_path)
                else:
                    print(f'File already exists: {dest_path}')


# def parsing_text_traj(text_traj):
#     pattern = re.search(
#         r'(?:(Your primary goal is to:.*?)\n(To achieve this,.*?)\n(Your task is to:.*?)\n(You are in the house,.*?)\n)|(?:(Your task is to:.*?)\n(You are in the house,.*?)\n)',
#         text_traj, re.DOTALL
#     )
    
#     if pattern.group(1):
#         parsing_result = {
#             'primary_goal': pattern.group(1).strip(),
#             'sibling_goals': pattern.group(2).strip(),
#             'task_goal': pattern.group(3).strip(),
#             'initial_state': pattern.group(4).strip()
#         }
#     else:
#         parsing_result = {
#             'task_goal': pattern.group(5).strip(),
#             'initial_state': pattern.group(6).strip()
#         }
#     return parsing_result

def parsing_text_traj(text_traj):
    task_goal_match = re.search(r"Your task is to: (.+)", text_traj)
    if not task_goal_match:
        raise ValueError("❌ 'Your task is to:' not found in trajectory.")
    
    task_goal = task_goal_match.group(1).strip()

    history_matches = re.findall(r"Your \d+ previous actions were: (.+)", text_traj)
    action_history = history_matches[-1].strip() if history_matches else ""

    return {
        'task_goal': task_goal,
        'action_history': action_history
    }

def embed_traj(cfg):
    em_dir = cfg.llm_agent.em_dir
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    
    success_traj_dir = os.path.join(em_dir, 'text_traj')
    embed_dir = os.path.join(em_dir, 'embed')
    os.makedirs(embed_dir, exist_ok=True)


    for file_name in os.listdir(success_traj_dir):
        file_path = os.path.join(success_traj_dir, file_name)
        
        with open(file_path, 'r') as file:
            text_traj = file.read()
        
        # parsing_result = parsing_text_traj(text_traj)
        
        parsed = parsing_text_traj(text_traj)
        task_goal = parsed['task_goal']
        action_history = parsed.get('action_history', '')

        task_embed = sbert.encode(task_goal)
        if action_history:
            history_embed = sbert.encode(action_history)
        else:
            history_embed = np.zeros_like(task_embed)
        goal_embed = 0.35 * task_embed + 0.65 * history_embed
        token_count = len(tokenizer(text_traj)['input_ids'])

        embedding = {
            'text_trajectory': text_traj,
            'embedding': goal_embed,
            'token_count': token_count,
            'text_traj_path': file_path,
            'task_goal': task_goal,
            'action_history': action_history
        }
        embed_name = file_name.replace('.txt', '.pkl')
        em_embed_path = os.path.join(embed_dir, embed_name)
        with open(em_embed_path, 'wb') as pickle_file:
            pickle.dump(embedding, pickle_file)









        # if token_count < 5000:
        #     with open(em_embed_path, 'wb') as pickle_file:
        #         pickle.dump(embedding, pickle_file)



@hydra.main(version_base=None, config_path='../conf', config_name='config_wah')
def main(cfg):
    # extract_success_traj(cfg)
    embed_traj(cfg)

if __name__ == "__main__":
    main()
