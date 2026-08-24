import os
import sys


from core.planner.base_planner import BasePlanner



class ReAct(BasePlanner):
    def __init__(self, cfg, env, llm_agent):
        super().__init__(cfg, env, llm_agent)
        self.max_decision_step = cfg.llm_agent.max_decision_step
        self.cur_decision_step = cfg.llm_agent.initial_step

    def run(self, task_data, log):
        self.cur_decision_step = 0
        init_obs = self.env.reset(task_data)
        target_info = self.make_target_info(task_data, init_obs)
        self.llm_agent.reset(target_info)
        self.initial_logging(log, task_data, init_obs)
        
        while True:
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                log.info('Max decisions step reached.')
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
                    if obs['success']:
                        self.llm_agent.add_text_traj(obs['obs_text']+'\n')
                    else:
                        self.llm_agent.add_text_traj(obs['feedback']+'\n')
                    
                    self.cur_decision_step +=1

                    log.info(obs['obs_text'])

    def collect_human(self, task_data, collect_dir):
        self.cur_decision_step = 0
        traj_file_path = os.path.join(collect_dir, f"traj_{task_data['env_id']}_{task_data['mode']}.txt")
        
        init_obs = self.env.reset(task_data)
        self.initial_collect(traj_file_path, task_data, init_obs)

        while True:
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                with open(traj_file_path, 'a') as f:
                    f.write('Max decision steps reached.')
                return terminate_info
            
            next_step_class = input('Type decision ("r" or "a"): ')
            skill_set = self.env.get_skill_set()

            if next_step_class == 'r':
                next_step = input('Think: ')
                self.cur_decision_step += 1
                with open(traj_file_path, 'a') as f:
                    f.write(f'Think: {next_step}\nOK.\n')
                pass
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
                    if obs['success']:
                        observation = obs['obs_text']
                    else:
                        observation = obs['feedback']
                    
                    self.llm_agent.add_text_traj(observation + '\n')
                    self.cur_decision_step +=1
                    self.write_text_traj(traj_file_path, observation)
            print(f'{self.cur_decision_step}: {next_step_class}: {next_step}')  

    def make_target_info(self):
        raise NotImplementedError("make_target_info() is not implemented.")
    
    def get_result(self, task_d):
        raise NotImplementedError()
