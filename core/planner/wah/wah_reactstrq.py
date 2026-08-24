import os
import sys


from core.planner.wah.reactstrq import ReActStrQ
import core.utils.wah_utils as wah_utils


class WahReActStrQ(ReActStrQ):
    def make_target_info(self, task_data, init_obs):
        target_info = {}
        target_info['nl_inst'] = task_data['nl_instructions'][0]
        target_info['init_obs_text'] = init_obs['obs_text']

        return target_info
    
    def initial_logging(self, log, task_data, init_obs):
        log.info(f'Task ID: {task_data["task_id"]}')
        log.info(f'Your task is to: {task_data["nl_instructions"][0]}')
        log.info(init_obs['obs_text'])

    def get_result(self, task_d):
        task_goal, graph = task_d['task_goal'], self.env.get_graph()
        name_id_dict_sim2nl, name_id_dict_nl2sim = self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim
        subgoal_success_rate = wah_utils.check_goal_condition(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim)
        if subgoal_success_rate == 1:
            goal_success_rate = 1
        else:
            goal_success_rate = 0

        result = {'task_id': task_d['task_id'],
                    'nl_inst': task_d['nl_instructions'][0],
                    'goal_success_rate': goal_success_rate,
                    'subgoal_success_rate': subgoal_success_rate}
        return result
    
    def initial_collect(self, traj_file_path, task_data, init_obs):
        with open(traj_file_path, 'w') as f:
            f.write(f'Your task is to: {task_data["nl_instructions"][0]}\n')
            f.write(init_obs['obs_text'] + '\n')

        print(f'Your task is to: {task_data["nl_instructions"][0]}\n')
        print(init_obs['obs_text'] + '\n')
    
    def write_text_traj(self, traj_file_path, text):
        with open(traj_file_path, 'a') as f:
            f.write(text + '\n')