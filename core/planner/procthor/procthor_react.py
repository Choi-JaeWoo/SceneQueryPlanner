import os
import sys


from core.planner.procthor.react import ReAct
import core.utils.procthor_utils as procthor_utils


class ProcThorReAct(ReAct):
    def make_target_info(self, task_data, init_obs):
        target_info = {}
        target_info['nl_inst'] = task_data['instruction'][0]
        target_info['init_obs_text'] = init_obs['obs_text']

        return target_info
    
    def initial_logging(self, log, task_data, init_obs):
        log.info(f'Task ID: {task_data["env_id"]}')
        log.info(f'Your task is to: {task_data["instruction"][0]}')
        log.info(init_obs['obs_text'])

    def get_result(self, task_d):
        task_goal = task_d['task_goal']
        subgoal_success_rate = procthor_utils.check_goal_condition(task_d, self.env.controller, self.env.init_event, self.env.cleaned_objects, self.env.cooled_objects, self.env.heated_objects, self.env.filled_coffee_objects)
        if subgoal_success_rate == 1:
            goal_success_rate = 1
        else:
            goal_success_rate = 0

        result = {'task_id': task_d['env_id'],
                    'nl_inst': task_d['instruction'][0],
                    'goal_success_rate': goal_success_rate,
                    'subgoal_success_rate': subgoal_success_rate}
        return result
    
    def initial_collect(self, traj_file_path, task_data, init_obs):
        with open(traj_file_path, 'w') as f:
            f.write(f'Your task is to: {task_data["instruction"][0]}\n')
            f.write(init_obs['obs_text'] + '\n')

        print(f'Your task is to: {task_data["instruction"][0]}\n')
        print(init_obs['obs_text'] + '\n')
    
    def write_text_traj(self, traj_file_path, text):
        with open(traj_file_path, 'a') as f:
            f.write(text + '\n')