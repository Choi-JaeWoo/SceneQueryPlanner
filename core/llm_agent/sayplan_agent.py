import os
import sys
import guidance
import torch


from core.llm_agent.base_agent import BaseAgent
from sentence_transformers import SentenceTransformer



class SayPlanAgent():
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

    def reset(self, target_info, planner_mode):
        """
        Reset LLM Agent with prompt.
        """
        # Load system prompt
        self.prompt = self.make_prompt(target_info, planner_mode)

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
    
    def decision_making(self, skill_set, planner_mode):
        """
        Decide the next step based on current prompt.
        """
        if planner_mode == 'semantic_search':
            if self.use_guidance:
                if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
                    raise NotImplementedError("OpenAI API is not supported yet.")
                else:
                    self.llm += guidance.select(['chain-of-thought: ', 'reasoning: ', 'command: '], name='choice')
                    if self.llm['choice'] == 'chain-of-thought: ':
                        self.llm += guidance.gen(stop='\n', name='cot', max_tokens=200, temperature=0) + '\n'
                        next_step_info = {'next_step_class': 'chain-of-thought', 'next_step': self.llm['cot']}
                    elif self.llm['choice'] == 'reasoning: ':
                        self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\n'
                        next_step_info = {'next_step_class': 'reasoning', 'next_step': self.llm['reasoning']}
                    else:
                        if self.tool_select_type == 'select':
                            self.llm += guidance.select(skill_set, name='command') + '\n'
                            next_step_info = {'next_step_class': 'command', 'next_step': self.llm['command']}
                        elif self.tool_select_type == 'generate':
                            raise NotImplementedError("Generate tool selection is not supported yet.")
                    return next_step_info
            else:
                raise NotImplementedError("Only guidance is supported for now.")
        elif planner_mode == 'iterative_replanning':
            if self.use_guidance:
                if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
                    raise NotImplementedError("OpenAI API is not supported yet.")
                else:
                    self.llm += guidance.select(['chain-of-thought: ', 'reasoning: ', 'plan: '], name='choice')
                    if self.llm['choice'] == 'chain-of-thought: ':
                        self.llm += guidance.gen(stop='\n', name='cot', max_tokens=200, temperature=0) + '\n'
                        next_step_info = {'next_step_class': 'chain-of-thought', 'next_step': self.llm['cot']}
                    elif self.llm['choice'] == 'reasoning: ':
                        self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\n'
                        next_step_info = {'next_step_class': 'reasoning', 'next_step': self.llm['reasoning']}
                    else:
                        self.llm += guidance.gen(stop='\n', name='plan', max_tokens=200, temperature=0) + '\n'
                        next_step_info = {'next_step_class': 'plan', 'next_step': self.llm['plan']}
                        # if self.tool_select_type == 'select':
                        #     self.llm += guidance.select(skill_set, name='plan') + '\n'
                        #     next_step_info = {'next_step_class': 'plan', 'next_step': self.llm['plan']}
                        # elif self.tool_select_type == 'generate':
                        #     raise NotImplementedError("Generate tool selection is not supported yet.")
                    return next_step_info
            else:
                raise NotImplementedError("Only guidance is supported for now.")
        else:
            raise NotImplementedError()

    def add_text_traj(self, text_traj):
        if self.use_guidance:
            if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
                raise NotImplementedError("OpenAI API is not supported yet.")
            else:
                self.llm += text_traj
        else:
            raise NotImplementedError("Only guidance is supported for now.")