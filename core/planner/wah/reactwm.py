import os
import sys


from core.planner.base_planner import BasePlanner
from core.wm.wm_wah import WahWM
from core.wm.sg_wah import WahSG
from core.wm.sg_wah import WahSG_One
from core.retriever.interface_wah import SceneGraphInterface, SceneGraphInterface_One
from core.utils.wah_utils import recall_working_memory, decompose_nl_skill


class ReActWM(BasePlanner):
    def __init__(self, cfg, env, llm_agent):
        super().__init__(cfg, env, llm_agent)
        self.max_decision_step = cfg.llm_agent.max_decision_step
        self.cur_decision_step = cfg.llm_agent.initial_step
        # self.retriever = retriever

    def run(self, task_data, log):
        self.cur_decision_step = 0

        init_obs = self.env.reset(task_data)
        
        partial_graph = self.env.get_graph_obs(visibility='initial')

        working_memory = WahWM(self.env)
        working_memory.update_graph(None)

        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info)


        ##########
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
                    # sim_graph = self.env.get_graph_obs(visibility='partial')
                    # if 'go to' in next_step:
                    #     working_memory.update_nx_graph(sim_graph, next_step.split('go to ')[-1])    
                    # else:
                    #     working_memory.update_nx_graph(sim_graph)
                    ###
                    sim_graph = self.env.get_graph_obs(visibility='partial')
                    sim_skill_info = decompose_nl_skill(next_step, self.env.name_id_dict_nl2sim, self.env.cur_recep_info)
                    working_memory.update_graph(sim_skill_info)
                    ###
                    if obs['success']:
                        observation = obs['obs_text']
                    else:
                        observation = obs['feedback']
                    
                    self.llm_agent.add_text_traj(observation + '\n')
                    self.cur_decision_step +=1
                    log.info(observation)
            elif next_step_class == 'Query':
                #####
                self.cur_decision_step += 1
                if 'recall location of' in next_step:
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = recall_working_memory(working_memory.scene_graph, target_obj)
                else:
                    obs_text = 'Error: Invalid query command.'
                self.llm_agent.add_text_traj(obs_text + '\n')
                log.info(obs_text)
                #####

    def collect_human(self, task_data, collect_dir):
        self.cur_decision_step = 0
        traj_file_path = os.path.join(collect_dir, f"traj_{task_data['task_id']:03d}.txt")
        
        init_obs = self.env.reset(task_data)
        self.initial_collect(traj_file_path, task_data, init_obs)

        partial_graph = self.env.get_graph_obs(visibility='initial')

        working_memory = WahWM(self.env)
        working_memory.update_graph(None)
        
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
                next_step = input('Location Query: ')
                self.cur_decision_step += 1
                #########
                with open(traj_file_path, 'a') as f:
                    f.write(f'Query: {next_step}\n')
                
                target_obj = next_step.split('recall location of ')[1]
                obs_text = recall_working_memory(working_memory.scene_graph, target_obj)
                print(obs_text)
                with open(traj_file_path, 'a') as f:
                    f.write(f'{obs_text}\n')
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
                        sim_skill_info = decompose_nl_skill(next_step, self.env.name_id_dict_nl2sim, self.env.cur_recep_info)
                        working_memory.update_graph(sim_skill_info)

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
        traj_file_path = os.path.join(collect_dir, f"traj_{task_data['task_id']:03d}.txt")

        init_obs = self.env.reset(task_data)
        self.initial_collect(traj_file_path, task_data, init_obs)
        
        partial_graph = self.env.get_graph_obs(visibility='initial')

        working_memory = WahWM(self.env)
        working_memory.update_graph(None)

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
                    # sim_graph = self.env.get_graph_obs(visibility='partial')
                    # if 'go to' in next_step:
                    #     working_memory.update_nx_graph(sim_graph, next_step.split('go to ')[-1])    
                    # else:
                    #     working_memory.update_nx_graph(sim_graph)
                    
                    ###
                    sim_graph = self.env.get_graph_obs(visibility='partial')
                    sim_skill_info = decompose_nl_skill(next_step, self.env.name_id_dict_nl2sim, self.env.cur_recep_info)
                    working_memory.update_graph(sim_skill_info)
                    ###
                    if obs['success']:
                        observation = obs['obs_text']
                    else:
                        observation = obs['feedback']
                    
                    self.llm_agent.add_text_traj(observation + '\n')
                    self.cur_decision_step +=1
                    self.write_text_traj(traj_file_path, observation)
            elif next_step_class == 'Query':
                self.cur_decision_step += 1
                if 'recall location of' in next_step:
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = recall_working_memory(working_memory.scene_graph, target_obj)
                else:
                    obs_text = 'Error: Invalid query command.'
                self.llm_agent.add_text_traj(obs_text + '\n')
                self.write_text_traj(traj_file_path, obs_text)

            print(f'{self.cur_decision_step}: {next_step_class}: {next_step}')
        

    def make_target_info(self):
        raise NotImplementedError("make_target_info() is not implemented.")
    
    def get_result(self, task_d):
        raise NotImplementedError()
    


