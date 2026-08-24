import os
import sys
import guidance
import torch


from core.llm_agent.base_agent import BaseAgent
from sentence_transformers import SentenceTransformer

import numpy as np
from numpy.linalg import norm
from guidance import system, user, assistant
def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))



class ReActStrQAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.llm_agent.model_name
        self.use_guidance = cfg.llm_agent.use_guidance
        self.tool_select_type = cfg.llm_agent.tool_select_type
        self.em_dir = cfg.llm_agent.em_dir
        self.sys_prompt_path = cfg.llm_agent.sys_prompt_path
        self.retrieval_type = cfg.llm_agent.retrieval_type
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        if self.use_guidance:
            if self.model_name in ['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13', 'gpt-4o-2024-11-20']:
                self.llm = guidance.models.OpenAI(self.model_name, api_key=cfg.llm_agent.openai_api_key)
            else:
                model_args = {'trust_remote_code': True, 'torch_dtype': torch.float16}
                if cfg.llm_agent.use_accelerate_device_map:
                    model_args['device_map'] = "auto"
                    if cfg.llm_agent.load_in_8bit:
                        model_args['load_in_8bit'] = True
                model_args['use_auth_token'] = cfg.llm_agent.hf_auth_token
                self.llm = guidance.models.Transformers(self.model_name, echo=False, **model_args)
        else:
            raise NotImplementedError("Only guidance is supported for now.")

    def reset(self, target_info):
        """
        Reset LLM Agent with prompt.
        """
        # Load system prompt
        self.prompt = self.make_prompt(target_info)

        # Reset LLM
        if self.use_guidance:
            if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13', 'gpt-4o-2024-11-20']:
                self.llm = guidance.models.OpenAI(self.model_name, api_key=self.cfg.llm_agent.openai_api_key)
                self.llm.reset()
                with guidance.user():
                    self.llm += self.prompt
            else:
                self.llm.reset()
                self.llm += self.prompt
        else:
            raise NotImplementedError("Only guidance is supported for now.")

    def decision_making(self, skill_set):
        """
        Decide the next step based on current prompt.
        """
        if self.use_guidance:
            if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13', 'gpt-4o-2024-11-20']:
                temp = self.llm + guidance.gen(stop='\n', name='next_step', max_tokens=200, temperature=0)
                next_step_sent = temp['next_step']
                if 'Think: ' in next_step_sent:
                    next_step = next_step_sent.split('Think: ')[1]
                    with assistant():
                        self.llm += next_step_sent
                    with user():
                        self.llm += 'OK.\n'
                    next_step_info = {'next_step_class': 'Think', 'next_step': next_step}
                elif 'Act: 'in next_step_sent:
                    next_step = next_step_sent.split('Act: ')[1]
                    if next_step in skill_set:
                        with assistant():
                            self.llm += next_step_sent
                        next_step_info = {'next_step_class': 'Act', 'next_step': next_step}
                    else:
                        generated_skill_emb = self.sbert.encode(next_step)
                        skill_set_emb = self.sbert.encode(skill_set)
                        
                        best_sim = -1
                        best_idx = 0
                        for i, skill_emb in enumerate(skill_set_emb):
                            sim = cosine_similarity(generated_skill_emb, skill_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_idx = i
                        best_skill = skill_set[best_idx]
                        
                        with assistant():
                            self.llm += f"Act: {best_skill}"
                        
                        next_step_info = {'next_step_class': 'Act', 'next_step': best_skill}
                elif 'Query: ' in next_step_sent:
                    next_step = next_step_sent.split('Query: ')[1]
                    with assistant():
                        self.llm += next_step_sent
                    next_step_info = {'next_step_class': 'Query', 'next_step': next_step}
                else:
                    raise NotImplementedError("Exeption")
                return next_step_info
            else:
                self.llm += guidance.select(['Act: ', 'Think: ', 'Query: '], name='choice')
                if self.llm['choice'] == 'Think: ':
                    self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\nOK.\n'
                    next_step_info = {'next_step_class': 'Think', 'next_step': self.llm['reasoning']}
                elif self.llm['choice'] == 'Query: ':
                    self.llm += guidance.gen(stop='\n', name='querying', max_tokens=200, temperature=0) + '\n'
                    next_step_info = {'next_step_class': 'Query', 'next_step': self.llm['querying']}
                else:
                    if self.tool_select_type == 'select':
                        self.llm += guidance.select(skill_set, name='nl_skill') + '\n'
                        next_step_info = {'next_step_class': 'Act', 'next_step': self.llm['nl_skill']}
                    elif self.tool_select_type == 'generate':
                        raise NotImplementedError("Generate tool selection is not supported yet.")
                return next_step_info
        else:
            raise NotImplementedError("Only guidance is supported for now.")
    
    def make_prompt(self, *args, **kwargs):
        """
        Make prompt for LLM.
        """
        raise NotImplementedError("Make prompt is not implemented yet.")
    
    def add_text_traj(self, text_traj):
        if self.use_guidance:
            if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13', 'gpt-4o-2024-11-20']:
                with user():
                    self.llm += text_traj
            else:
                self.llm += text_traj
        else:
            raise NotImplementedError("Only guidance is supported for now.")