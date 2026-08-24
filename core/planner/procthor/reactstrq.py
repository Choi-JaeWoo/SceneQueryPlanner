import os
import sys


from core.planner.base_planner import BasePlanner
from core.wm.sg_procthor import ProcThorSG
from core.wm.sg_procthor import ProcThorSG_One
from core.wm.sg_procthor import draw_scene_graph


class ReActStrQ(BasePlanner):
    def __init__(self, cfg, env, llm_agent, retriever):
        super().__init__(cfg, env, llm_agent)
        self.max_decision_step = cfg.llm_agent.max_decision_step
        self.cur_decision_step = cfg.llm_agent.initial_step
        self.retriever = retriever

    def run(self, task_data, log):
        self.cur_decision_step = 0

        init_obs = self.env.reset(task_data)
        
        partial_graph = self.env.get_graph_obs(visibility='initial')
        working_memory = ProcThorSG_One(self.cfg, partial_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim)

        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info)

        self.initial_logging(log, task_data, init_obs)

        while True:
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                log.info('Max decision steps reached.')
                return terminate_info
            
            skill_set = self.env.get_skill_set()

            try:
                next_step_info = self.llm_agent.decision_making(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                log.info(f'{next_step_class}: {next_step}')
                
            except Exception as e:
                terminate_info = {'terminate': 'plan_next_step_error', 'decision_step': self.cur_decision_step}
                log.info(f"Plan Next Step Error: {e}")
                return terminate_info
            
            if next_step_class == 'Think':
                self.cur_decision_step += 1
                log.info('OK.')
                pass
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'decision_step': self.cur_decision_step}
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'decision_step': self.cur_decision_step}
                    return terminate_info
                else:
                    obs = self.env.step(next_step)
                    sim_graph = self.env.get_graph_obs(visibility='partial')
                    if 'go to' in next_step:
                        working_memory.update_nx_graph(sim_graph, self.env.last_event, next_step.split('go to ')[-1])    
                    else:
                        working_memory.update_nx_graph(sim_graph, self.env.last_event)
                    if obs['success']:
                        observation = obs['obs_text']
                    else:
                        observation = obs['feedback']
                    
                    self.llm_agent.add_text_traj(observation + '\n')
                    self.cur_decision_step +=1
                    log.info(observation)
            elif next_step_class == 'Query':
                current_graph = working_memory.scene_graph
                
                parsed_call = self.retriever._parse_interface_call_string(next_step)
                execution_outcome = self.retriever._execute_interface_call_from_parsed(parsed_call, current_graph)

                if execution_outcome['execution_successful']:
                    result_val = execution_outcome["result"]
                    if isinstance(result_val, list):
                        if all(isinstance(item, (str, int, float)) for item in result_val):
                            formatted_display_val = ", ".join(map(str, result_val))
                        else:
                            formatted_display_val = str(result_val)
                    else:
                        formatted_display_val = str(result_val)
                    retrieved_info = f'Info: {formatted_display_val}'

                    obs_vis = self.env.get_visual_obs()[0]
                    self.env.vis_log.append({'action': next_step, 'images': obs_vis, 'observation': formatted_display_val})

                else:
                    retrieved_info = f'Error: {execution_outcome["error"]}'
                self.cur_decision_step +=1
                self.llm_agent.add_text_traj(retrieved_info + '\n')
                log.info(retrieved_info)

    def collect_human(self, task_data, collect_dir):
        self.cur_decision_step = 0
        traj_file_path = os.path.join(collect_dir, f"traj_{task_data['env_id']}_{task_data['mode']}.txt")
        
        init_obs = self.env.reset(task_data)
        self.initial_collect(traj_file_path, task_data, init_obs)

        # partial_graph = self.env.get_graph_obs(visibility='partial')
        partial_graph = self.env.get_graph_obs(visibility='initial')
        # partial_graph = self.env.get_graph_obs(visibility='agent-centric')
        # working_memory = WahSG(self.cfg, partial_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim)
        working_memory = ProcThorSG_One(self.cfg, partial_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim)

        while True:
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                with open(traj_file_path, 'a') as f:
                    f.write('Max decision steps reached.')
                return terminate_info
            
            next_step_class = input('Type decision ("r" or "a" or "q"): ')
            skill_set = self.env.get_skill_set()

            if next_step_class == 'r':
                next_step = input('Think: ')
                self.cur_decision_step += 1
                with open(traj_file_path, 'a') as f:
                    f.write(f'Think: {next_step}\nOK.\n')
                pass
            elif next_step_class == 'q':
                next_step = input('Structured Query: ')
                self.cur_decision_step += 1
                current_graph = working_memory.scene_graph
                #########
                parsed_call = self.retriever._parse_interface_call_string(next_step)
                
                with open(traj_file_path, 'a') as f:
                    f.write(f'Query: {next_step}\n')
                
                execution_outcome = self.retriever._execute_interface_call_from_parsed(parsed_call, current_graph)

                if execution_outcome['execution_successful']:
                    result_val = execution_outcome["result"]
                    if isinstance(result_val, list):
                        if all(isinstance(item, (str, int, float)) for item in result_val):
                            formatted_display_val = ", ".join(map(str, result_val))
                        else:
                            formatted_display_val = str(result_val)
                    else:
                        formatted_display_val = str(result_val)


                    with open(traj_file_path, 'a') as f:
                        f.write(f'Info: {formatted_display_val}\n')
                    print(f'Info: {formatted_display_val}')
                else:
                    with open(traj_file_path, 'a') as f:
                        f.write(f'Error: {execution_outcome["error"]}\n')
                    print(f'Error: {execution_outcome["error"]}')
                #########
                
            elif next_step_class == 'a':
                next_step = input('Act: ')
                if next_step in skill_set:
                    with open(traj_file_path, 'a') as f:
                        f.write(f'Act: {next_step}\n')
                    if next_step == 'done':
                        terminate_info = {'terminate': 'done', 'decision_step': self.cur_decision_step}
                        return terminate_info
                    elif next_step == 'failure':
                        terminate_info = {'terminate': 'failure', 'decision_step': self.cur_decision_step}
                        return terminate_info
                    else:
                        obs = self.env.step(next_step)
                        sim_graph = self.env.get_graph_obs(visibility='partial')
                        if 'go to' in next_step:
                            working_memory.update_nx_graph(sim_graph, self.env.last_event, next_step.split('go to ')[-1])    
                        else:
                            working_memory.update_nx_graph(sim_graph, self.env.last_event)
                        if obs['success']:
                            observation = obs['obs_text']
                        else:
                            observation = obs['feedback']
                        self.cur_decision_step +=1
                        with open(traj_file_path, 'a') as f:
                            f.write(f'{observation}\n')
                        print(observation)
                else:
                    print(f"Invalid action: {next_step}. Please choose from {skill_set}.")
            
    def collect_llm(self, task_data, collect_dir):
        self.cur_decision_step = 0
        traj_file_path = os.path.join(collect_dir, f"traj_{task_data['env_id']}_{task_data['mode']}.txt")

        init_obs = self.env.reset(task_data)
        self.initial_collect(traj_file_path, task_data, init_obs)
        
        partial_graph = self.env.get_graph_obs(visibility='initial')
        working_memory = ProcThorSG_One(self.cfg, partial_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim)

        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info)

        while True:
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                self.write_text_traj(traj_file_path, 'Max decision steps reached.')
                return terminate_info
            
            skill_set = self.env.get_skill_set()

            try:
                next_step_info = self.llm_agent.decision_making(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                self.write_text_traj(traj_file_path, f'{next_step_class}: {next_step}')
            except Exception as e:
                terminate_info = {'terminate': 'plan_next_step_error', 'decision_step': self.cur_decision_step}
                self.write_text_traj(traj_file_path, f"Plan Next Step Error: {e}")
                return terminate_info
            if next_step_class == 'Think':
                self.cur_decision_step += 1
                self.write_text_traj(traj_file_path, 'OK.')
                pass
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'decision_step': self.cur_decision_step}
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'decision_step': self.cur_decision_step}
                    return terminate_info
                else:
                    obs = self.env.step(next_step)
                    sim_graph = self.env.get_graph_obs(visibility='partial')
                    if 'go to' in next_step:
                        working_memory.update_nx_graph(sim_graph, self.env.last_event, next_step.split('go to ')[-1])    
                    else:
                        working_memory.update_nx_graph(sim_graph, self.env.last_event)
                    if obs['success']:
                        observation = obs['obs_text']
                    else:
                        observation = obs['feedback']
                    
                    self.llm_agent.add_text_traj(observation + '\n')
                    self.cur_decision_step +=1
                    self.write_text_traj(traj_file_path, observation)
            elif next_step_class == 'Query':
                current_graph = working_memory.scene_graph
                
                parsed_call = self.retriever._parse_interface_call_string(next_step)
                execution_outcome = self.retriever._execute_interface_call_from_parsed(parsed_call, current_graph)

                if execution_outcome['execution_successful']:
                    result_val = execution_outcome["result"]
                    if isinstance(result_val, list):
                        if all(isinstance(item, (str, int, float)) for item in result_val):
                            formatted_display_val = ", ".join(map(str, result_val))
                        else:
                            formatted_display_val = str(result_val)
                    else:
                        formatted_display_val = str(result_val)
                    retrieved_info = f'Info: {formatted_display_val}'
                else:
                    retrieved_info = f'Error: {execution_outcome["error"]}'
                self.cur_decision_step +=1
                self.llm_agent.add_text_traj(retrieved_info + '\n')
                self.write_text_traj(traj_file_path, retrieved_info)

            print(f'{self.cur_decision_step}: {next_step_class}: {next_step}')
        

    def make_target_info(self):
        raise NotImplementedError("make_target_info() is not implemented.")
    
    def get_result(self, task_d):
        raise NotImplementedError()
    


