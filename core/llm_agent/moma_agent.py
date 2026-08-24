import os
import sys
import guidance
import torch
from collections import defaultdict, Counter


from core.llm_agent.base_agent import BaseAgent
from sentence_transformers import SentenceTransformer


def read_txt_file(file_path):
    with open(file_path) as file:
        txt_content = file.read()
    return txt_content

class MoMaAgent(BaseAgent):
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
            if self.model_name in ['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
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

    def reset(self, target_info, action_history):
        """
        Reset LLM Agent with prompt.
        """
        # Load system prompt
        self.prompt = self.make_prompt(target_info, action_history)

        # Reset LLM
        if self.use_guidance:
            if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
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
        self.llm += "Think: "
        self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\n'
        reasoning = self.llm['reasoning']
        self.llm += "Act: "
        self.llm += guidance.select(skill_set, name='nl_skill') + '\n'
        next_step_info = {'next_step_class': 'Act', 'next_step': self.llm['nl_skill'], 'reasoning': reasoning}
        return next_step_info

    def make_prompt(self, target_info):
        """
        Make prompt for LLM.
        """
        raise NotImplementedError("Make prompt is not implemented yet.")
        
    def add_text_traj(self, text_traj):
        if self.use_guidance:
            if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
                raise NotImplementedError("OpenAI API is not supported yet.")
            else:
                self.llm += text_traj
        else:
            raise NotImplementedError("Only guidance is supported for now.")