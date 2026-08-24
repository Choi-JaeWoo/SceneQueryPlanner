import os
import sys


from core.planner.wah.moma import MoMa
import core.utils.wah_utils as wah_utils




class WahMoMa(MoMa):
    def make_target_info(self, task_data, init_obs):
        target_info = {}
        target_info['nl_inst'] = task_data['nl_instructions'][0]
        target_info['init_obs_text'] = init_obs['obs_text']

        return target_info
    
    def initial_logging(self, log, task_data):
        log.info(f'Task ID: {task_data["env_id"]}')
        log.info(f'Your task is to: {task_data["nl_instructions"][0]}')

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
    
    def initial_collect(self, traj_file_path, task_data):
        with open(traj_file_path, 'w') as f:
            f.write(f'Your task is to: {task_data["nl_instructions"][0]}\n')

        print(f'Your task is to: {task_data["nl_instructions"][0]}\n')
    
    def write_text_traj(self, traj_file_path, text):
        with open(traj_file_path, 'a') as f:
            f.write(text)