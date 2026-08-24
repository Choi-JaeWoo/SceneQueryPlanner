import os
import sys


from core.planner.base_planner import BasePlanner
from core.wm.full_graph_wah import WahFullGraph
import core.utils.wah_utils as wah_utils


class SayPlan(BasePlanner):
    def __init__(self, cfg, env, llm_agent):
        super().__init__(cfg, env, llm_agent)
        self.max_decision_step = cfg.llm_agent.max_decision_step
        self.cur_decision_step = cfg.llm_agent.initial_step
    
    def run(self, task_data, log):
        self.sem_cur_decision_step = 0
        self.ite_cur_decision_step = 0
        
        init_obs = self.env.reset(task_data)
        
        full_graph = self.env.get_graph_obs(visibility='full')
        filtered_sim_objs = self.env.filtered_sim_objs        
        full_graph = wah_utils.extract_graph_by_class_names(full_graph, filtered_sim_objs)
        working_memory = WahFullGraph(self.cfg, full_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim, task_data)
        
        self.initial_logging(log, task_data, init_obs)
        
        ### Semantic Search ###
        sayplan_dict = working_memory.collapse_graph()
        
        self.initial_logging_sem(log, task_data, sayplan_dict)
        semantic_search_memory = []
        
        init_obs['sayplan_dict'] = sayplan_dict
        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info, 'semantic_search')
        
        search_num = 0
        max_search_num = 30
        
        while True:
            if search_num > max_search_num:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                log.info('Max decision steps reached.')
                break
            
            skill_set = self.get_semantic_skill_set(working_memory.scene_graph)

            try:
                next_step_info = self.llm_agent.decision_making(skill_set, 'semantic_search')
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                log.info(f'{next_step_class}: {next_step}')
                search_num += 1
            except Exception as e:
                terminate_info = {'terminate': 'semantic_search_error', 'decision_step': self.cur_decision_step}
                log.info(f"Semantic Search Error: {e}")
                search_num += 1
            
            if next_step_class == 'command':
                command = next_step

                if command == 'done':
                    terminate_info = {'terminate': 'done', 'decision_step': self.cur_decision_step}
                    break
                elif 'expand' in command:
                    node_id = command.split('expand')[1].replace('(','').replace(')', '')
                    semantic_search_memory.append(node_id)
                    sayplan_dict = working_memory.expand_node(node_id)
                    sg_custom, memory = self.log_semantic_search(log, sayplan_dict, semantic_search_memory)
                    self.llm_agent.add_text_traj(f'3D Scene Graph: {sg_custom}\nMemory: {memory}\n')

                elif 'contract' in command:
                    node_id = command.split('contract')[1].replace('(','').replace(')', '')
                    sayplan_dict = working_memory.contract_node(node_id)
                    sg_custom, memory = self.log_semantic_search(log, sayplan_dict, semantic_search_memory)
                    self.llm_agent.add_text_traj(f'3D Scene Graph: {sg_custom}\nMemory: {memory}\n')
                else:
                    print("Invalid command. Please use 'expand(node_id)', 'contract(node_id)', or 'done'.")

        ### Iterative Replanning
        self.initial_logging_iter(log, task_data, sayplan_dict, semantic_search_memory)
        ######################################

        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info, 'iterative_replanning')
        
        # SKILL SET
        skill_set = self.get_iterative_skill_set(self.env)
        replan_num = 0
        max_replan_num = 60

        while True:
            if replan_num > max_replan_num:
                log.info('Max decision steps reached.')
                break
            try:
                next_step_info = self.llm_agent.decision_making(skill_set, 'iterative_replanning')
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                log.info(f'{next_step_class}: {next_step}')
                replan_num += 1
            except Exception as e:
                terminate_info = {'terminate': 'iterative_replanning_error', 'decision_step': self.cur_decision_step}
                log.info(f"Iterative Replanning Error: {e}")
                replan_num += 1
            
            if next_step_class == 'plan':
                feedback = self.simulate_plan(next_step, task_data, skill_set)
                if feedback == 'done':
                    log.info('Scene Graph Simulator: Plan Verified')
                    break
                else:
                    log.info(f'Scene Graph Simulator: {feedback}')
                    self.llm_agent.add_text_traj(f'Scene Graph Simulator: {feedback}\n')

    
    def collect_human(self, task_data, collect_dir):
        self.sem_cur_decision_step = 0
        self.ite_cur_decision_step = 0
        semantic_search_dir = os.path.join(collect_dir, 'semantic_search')
        iterative_replanning_dir = os.path.join(collect_dir, 'iterative_replanning')

        os.makedirs(semantic_search_dir, exist_ok=True)
        os.makedirs(iterative_replanning_dir, exist_ok=True)

        init_obs = self.env.reset(task_data)
        
        full_graph = self.env.get_graph_obs(visibility='full')
        filtered_sim_objs = self.env.filtered_sim_objs        
        full_graph = wah_utils.extract_graph_by_class_names(full_graph, filtered_sim_objs)
        working_memory = WahFullGraph(self.cfg, full_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim, task_data)
        
        ### Semantic Search
        semantic_search_path = os.path.join(semantic_search_dir, f'{str(task_data["task_id"]).zfill(3)}_semantic_search.txt')
        sayplan_dict = working_memory.collapse_graph()
        
        self.initial_collect_semantic_search(semantic_search_path, task_data, sayplan_dict)
        semantic_search_memory = []
        while True:
            if self.sem_cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                with open(semantic_search_path, 'a') as f:
                    f.write('Max decision steps reached.')
                break
            
            chain_of_thought = input('chain-of-thought: ')
            reasoning = input('reasoning: ')
            with open(semantic_search_path, 'a') as f:
                f.write(f'chain-of-thought: {chain_of_thought}\n')
                f.write(f'reasoning: {reasoning}\n')
            
            command = input('command: ')
            if command == 'done':
                with open(semantic_search_path, 'a') as f:
                    f.write(f'command: {command}')
                terminate_info = {'terminate': 'done', 'decision_step': self.cur_decision_step}
                break
            elif command == 'failure':
                with open(semantic_search_path, 'a') as f:
                    f.write(f'command: {command}')
                terminate_info = {'terminate': 'failure', 'decision_step': self.cur_decision_step}
                break
            elif 'expand' in command:
                node_id = command.split('expand')[1].replace('(','').replace(')', '')
                semantic_search_memory.append(node_id)
                sayplan_dict = working_memory.expand_node(node_id)
                with open(semantic_search_path, 'a') as f:
                    f.write(f'command: {command}\n')
                self.write_semantic_search(semantic_search_path, sayplan_dict, semantic_search_memory)
            elif 'contract' in command:
                node_id = command.split('contract')[1].replace('(','').replace(')', '')
                sayplan_dict = working_memory.contract_node(node_id)
                with open(semantic_search_path, 'a') as f:
                    f.write(f'command: {command}\n')
                self.write_semantic_search(semantic_search_path, sayplan_dict, semantic_search_memory)
            else:
                print("Invalid command. Please use 'expand(node_id)', 'contract(node_id)', or 'done'.")
        
        ### Iterative Replanning
        iterative_replanning_path = os.path.join(iterative_replanning_dir, f'{str(task_data["task_id"]).zfill(3)}_iterative_replanning.txt')
        self.initial_collect_iterative_replanning(iterative_replanning_path, task_data, sayplan_dict, semantic_search_memory)
        
        while True:
            chain_of_thought = input('chain-of-thought: ')
            reasoning = input('reasoning: ')
            with open(iterative_replanning_path, 'a') as f:
                f.write(f'chain-of-thought: {chain_of_thought}\n')
                f.write(f'reasoning: {reasoning}\n')

            plan = input('plan: ')
            with open(iterative_replanning_path, 'a') as f:
                f.write(f'plan: [{plan}]\n')
            feedback = self.simulate_plan(f'[{plan}]', task_data)
            print((f'Scene Graph Simulator: {feedback}\n'))
            if feedback == 'done':
                with open(iterative_replanning_path, 'a') as f:
                    f.write('Scene Graph Simulator: Plan Verified\n')
                break
            else:
                with open(iterative_replanning_path, 'a') as f:
                    f.write(f'Scene Graph Simulator: {feedback}\n')

    def collect_llm(self, task_data, collect_dir):
        self.sem_cur_decision_step = 0
        self.ite_cur_decision_step = 0
        semantic_search_dir = os.path.join(collect_dir, 'semantic_search')
        iterative_replanning_dir = os.path.join(collect_dir, 'iterative_replanning')

        os.makedirs(semantic_search_dir, exist_ok=True)
        os.makedirs(iterative_replanning_dir, exist_ok=True)

        init_obs = self.env.reset(task_data)
        
        full_graph = self.env.get_graph_obs(visibility='full')
        filtered_sim_objs = self.env.filtered_sim_objs        
        full_graph = wah_utils.extract_graph_by_class_names(full_graph, filtered_sim_objs)
        working_memory = WahFullGraph(self.cfg, full_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim, task_data)
        
        ### Semantic Search ###
        semantic_search_path = os.path.join(semantic_search_dir, f'{str(task_data["task_id"]).zfill(3)}_semantic_search.txt')
        sayplan_dict = working_memory.collapse_graph()
        
        self.initial_collect_semantic_search(semantic_search_path, task_data, sayplan_dict)
        semantic_search_memory = []
        
        init_obs['sayplan_dict'] = sayplan_dict
        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info, 'semantic_search')
        
        search_num = 0
        max_search_num = 30
        while True:
            if search_num > max_search_num:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                self.write_text_traj(semantic_search_path, 'Max decision steps reached.')
                break
            
            skill_set = self.get_semantic_skill_set(working_memory.scene_graph)

            try:
                next_step_info = self.llm_agent.decision_making(skill_set, 'semantic_search')
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                self.write_text_traj(semantic_search_path, f'{next_step_class}: {next_step}')
                search_num += 1
            except Exception as e:
                terminate_info = {'terminate': 'semantic_search_error', 'decision_step': self.cur_decision_step}
                self.write_text_traj(semantic_search_path, f"Semantic Search Error: {e}")
                search_num += 1
            
            if next_step_class == 'command':
                command = next_step

                if command == 'done':
                    terminate_info = {'terminate': 'done', 'decision_step': self.cur_decision_step}
                    break
                elif 'expand' in command:
                    node_id = command.split('expand')[1].replace('(','').replace(')', '')
                    semantic_search_memory.append(node_id)
                    sayplan_dict = working_memory.expand_node(node_id)
                    sg_custom, memory = self.write_semantic_search(semantic_search_path, sayplan_dict, semantic_search_memory)
                    self.llm_agent.add_text_traj(f'3D Scene Graph: {sg_custom}\nMemory: {memory}\n')

                elif 'contract' in command:
                    node_id = command.split('contract')[1].replace('(','').replace(')', '')
                    sayplan_dict = working_memory.contract_node(node_id)
                    sg_custom, memory = self.write_semantic_search(semantic_search_path, sayplan_dict, semantic_search_memory)
                    self.llm_agent.add_text_traj(f'3D Scene Graph: {sg_custom}\nMemory: {memory}\n')
                else:
                    print("Invalid command. Please use 'expand(node_id)', 'contract(node_id)', or 'done'.")

        ### Iterative Replanning
        iterative_replanning_path = os.path.join(iterative_replanning_dir, f'{str(task_data["task_id"]).zfill(3)}_iterative_replanning.txt')
        self.initial_collect_iterative_replanning(iterative_replanning_path, task_data, sayplan_dict, semantic_search_memory)
        ######################################

        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info, 'iterative_replanning')
        
        # SKILL SET
        skill_set = self.get_iterative_skill_set(self.env)
        replan_num = 0
        max_replan_num = 60

        while True:
            if replan_num > max_replan_num:
                self.write_text_traj(iterative_replanning_path, 'Max decision steps reached.')
                break
            try:
                next_step_info = self.llm_agent.decision_making(skill_set, 'iterative_replanning')
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                self.write_text_traj(iterative_replanning_path, f'{next_step_class}: {next_step}')
                print(f'{next_step_class}: {next_step}')
                replan_num += 1
            except Exception as e:
                terminate_info = {'terminate': 'iterative_replanning_error', 'decision_step': self.cur_decision_step}
                self.write_text_traj(iterative_replanning_path, f"Iterative Replanning Error: {e}")
                replan_num += 1
            
            if next_step_class == 'plan':
                feedback = self.simulate_plan(next_step, task_data, skill_set)
                print(feedback)
                if feedback == 'done':
                    self.write_text_traj(iterative_replanning_path, 'Scene Graph Simulator: Plan Verified')
                    break
                else:
                    self.write_text_traj(iterative_replanning_path, f'Scene Graph Simulator: {feedback}')
                    self.llm_agent.add_text_traj(f'Scene Graph Simulator: {feedback}\n')