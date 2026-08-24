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
    
    for file_name in file_names:
        file_path = os.path.join(collect_traj_dir, file_name)
        task_id = get_task_id_from_file(file_path)
        task_d = train_set[task_id]

        env.reset(task_d)
        act_seq = extract_act_seq(file_path)

        for nl_action in act_seq:
            if nl_action in ['done', 'failure']:
                pass
            else:
                env.step(nl_action)


        ssr = check_goal_condition(task_d['task_goal'], env.get_graph(), env.name_id_dict_sim2nl, env.name_id_dict_nl2sim)
        if ssr == 1:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(success_traj_dir, filename)
            if not os.path.exists(dest_path):
                shutil.copy(file_path, dest_path)
            else:
                print('File already exists.')


def parsing_text_traj(text_traj):
    pattern = re.search(
        r'(?:(Your primary goal is to:.*?)\n(To achieve this,.*?)\n(Your task is to:.*?)\n(You are in the house,.*?)\n)|(?:(Your task is to:.*?)\n(You are in the house,.*?)\n)',
        text_traj, re.DOTALL
    )
    
    if pattern.group(1):
        parsing_result = {
            'primary_goal': pattern.group(1).strip(),
            'sibling_goals': pattern.group(2).strip(),
            'task_goal': pattern.group(3).strip(),
            'initial_state': pattern.group(4).strip()
        }
    else:
        parsing_result = {
            'task_goal': pattern.group(5).strip(),
            'initial_state': pattern.group(6).strip()
        }
    return parsing_result

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
        
        parsing_result = parsing_text_traj(text_traj)
        
        task_goal_text = parsing_result['task_goal']
        goal_embedding = sbert.encode(task_goal_text.split('Your task is to: ')[1])
        
        tokens = tokenizer(text_traj)['input_ids']
        token_count = len(tokens)
        
        state_embedding = sbert.encode(parsing_result['initial_state'])

        embed_name = file_name.replace('.txt', '.pkl')
        embedding = {'text_trajectory': text_traj,
            'embedding': goal_embedding,
            'text_traj_path': file_path,
            'token_count': token_count,
            'state_embedding': state_embedding}
        em_embed_path = os.path.join(embed_dir, embed_name)
        with open(em_embed_path, 'wb') as pickle_file:
            pickle.dump(embedding, pickle_file)
        # if token_count < 5000:
        #     with open(em_embed_path, 'wb') as pickle_file:
        #         pickle.dump(embedding, pickle_file)



@hydra.main(version_base=None, config_path='../conf', config_name='config_wah')
def main(cfg):
    extract_success_traj(cfg)
    embed_traj(cfg)
    

if __name__ == "__main__":
    main()
