import os
import sys
import pickle
import numpy as np


from core.llm_agent.moma_agent import MoMaAgent
from numpy.linalg import norm
from transformers import AutoTokenizer
import core.utils.procthor_utils as procthor_utils


def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

def read_txt_file(file_path):
    with open(file_path) as file:
        txt_content = file.read()
    return txt_content

class ProcThorMoMaAgent(MoMaAgent):        
    def make_prompt(self, target_info, action_history):
        nl_inst = target_info['nl_inst']
        
        # Load system prompt
        sys_prompt = read_txt_file(self.sys_prompt_path)
        
        # Load in-context examples
        ic_ex_files = self.retrieve_exs(target_info, action_history)
        ic_exs = [read_txt_file(ic_ex_file) for ic_ex_file in ic_ex_files]
        ic_ex_prompt = '\n'.join(ic_ex for ic_ex in ic_exs)

        prompt = f'{sys_prompt}\n\nSource domain:\n{ic_ex_prompt}\nTarget_domain:\nYour task is to: {nl_inst}\n'
        return prompt
    
    def retrieve_exs(self, target_info, action_history):
        retrieval_type = self.retrieval_type
        if retrieval_type == 'similarity':
            ic_ex_embedding_dir = os.path.join(self.em_dir, 'embed')
            sbert = self.sbert
            target_nl_inst = target_info['nl_inst']
            inst_emb = sbert.encode(target_nl_inst)            
            if action_history:
                formatted_history = ', '.join([
                    f"{step} (success)" if success else f"{step} (fail)" 
                    for step, success in action_history
                ])
                hist_emb = sbert.encode(formatted_history)
            else:
                hist_emb = np.zeros_like(inst_emb)
            alpha, beta = 0.35, 0.65
            target_nl_inst_embedding = alpha * inst_emb + beta * hist_emb
            
            # Load all embeddings
            ic_ex_embedding_list = []
            for ic_ex_embedding_name in os.listdir(ic_ex_embedding_dir):
                if ic_ex_embedding_name.endswith('.pkl'):
                    ic_ex_embedding_path = os.path.join(ic_ex_embedding_dir, ic_ex_embedding_name)

                    with open(ic_ex_embedding_path, 'rb') as file:
                        ic_ex_embedding = pickle.load(file)

                    # Resolve the trajectory text file relative to em_dir so the
                    # example bank can be relocated without re-embedding.
                    ic_ex_embedding['text_traj_path'] = os.path.join(
                        os.path.dirname(ic_ex_embedding_dir), 'text_traj',
                        ic_ex_embedding_name.replace('.pkl', '.txt'))
                    
                    similarity = cosine_similarity(ic_ex_embedding['embedding'], target_nl_inst_embedding)

                    ic_ex_embedding['similarity'] = similarity
                    ic_ex_embedding_list.append(ic_ex_embedding)
            sorted_ic_ex_embedding_list = sorted(ic_ex_embedding_list, key=lambda x: x['similarity'], reverse=True)
            sorted_ic_ex_embedding_list = procthor_utils.sort_with_same_similarity(sorted_ic_ex_embedding_list)
            ic_ex_files = []
            current_token_sum = 0
            for embedding in sorted_ic_ex_embedding_list:
                token_count = embedding['token_count']
                if token_count > 5000:
                    continue
                if current_token_sum + token_count > 5000:
                    break
                ic_ex_files.append(embedding['text_traj_path'])
                current_token_sum += token_count
        else:
            raise NotImplementedError("Only similarity retrieval is supported for now.")
        return ic_ex_files
