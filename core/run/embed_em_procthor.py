import os, sys, json, shutil, re, pickle, hydra, logging
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from core.env.procthor_env import ProcThorEnv
from core.utils.procthor_utils import check_goal_condition

log = logging.getLogger(__name__)


def parse_filename(file_name):
    match = re.match(r"traj_(\d+)_(\w+)\.txt", file_name)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None


def load_taskset(eval_path):
    with open(eval_path, 'r') as f:
        return json.load(f)


def extract_act_seq(file_path):
    act_seq = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Act: "):
                act_seq.append(line.strip().split("Act: ")[1])
    return act_seq


def parsing_text_traj(text_traj):
    pattern = re.search(
        r'(?:(Your primary goal is to:.*?)\n(To achieve this,.*?)\n(Your task is to:.*?)\n(You are in the house,.*?)\n)|(?:(Your task is to:.*?)\n(You are in the house,.*?)\n)',
        text_traj, re.DOTALL
    )
    if pattern.group(1):
        return {
            'primary_goal': pattern.group(1).strip(),
            'sibling_goals': pattern.group(2).strip(),
            'task_goal': pattern.group(3).strip(),
            'initial_state': pattern.group(4).strip()
        }
    else:
        return {
            'task_goal': pattern.group(5).strip(),
            'initial_state': pattern.group(6).strip()
        }


def extract_success_traj(cfg, task_list):
    env = ProcThorEnv(cfg)
    collect_dir = cfg.dataset.collect_dir
    save_dir = os.path.join(cfg.llm_agent.em_dir, "text_traj")
    os.makedirs(save_dir, exist_ok=True)

    for file_name in os.listdir(collect_dir):
        file_path = os.path.join(collect_dir, file_name)
        env_id, task_type = parse_filename(file_name)
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

        env.reset(task_d)
        act_seq = extract_act_seq(file_path)
        for act in act_seq:
            if act not in ['done', 'failure']:
                env.step(act)

        ssr = check_goal_condition(
            task_d, env.controller, env.init_event,
            env.cleaned_objects, env.cooled_objects,
            env.heated_objects, env.filled_coffee_objects
        )
        print(f"[INFO] Subgoal Success Rate (SSR): {ssr}")

        if ssr == 1:
            shutil.copy(file_path, os.path.join(save_dir, file_name))


def embed_traj(cfg):
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    input_dir = os.path.join(cfg.llm_agent.em_dir, 'text_traj')
    output_dir = os.path.join(cfg.llm_agent.em_dir, 'embed')
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r') as f:
            text = f.read()

        parsed = parsing_text_traj(text)
        goal_embed = sbert.encode(parsed['task_goal'].split("Your task is to: ")[-1])
        token_count = len(tokenizer(text)['input_ids'])

        embed_obj = {
            'text_trajectory': text,
            'embedding': goal_embed,
            'token_count': token_count,
            'text_traj_path': file_path
        }

        with open(os.path.join(output_dir, file_name.replace(".txt", ".pkl")), 'wb') as f:
            pickle.dump(embed_obj, f)
            
        count += 1
        
    print(f"✅ Total embeddings created: {count}")

@hydra.main(version_base=None, config_path="../conf", config_name="config_procthor")
def main(cfg):
    eval_path = cfg.procthor.eval_set
    task_list = load_taskset(eval_path)
    print("🚀 Extracting successful trajectories...")
    extract_success_traj(cfg, task_list)
    print("📌 Embedding successful trajectories...")
    embed_traj(cfg)
    print("✅ All done.")


if __name__ == "__main__":
    main()
