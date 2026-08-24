import os
import sys
import json
import re


from core.planner.wah.sayplan import SayPlan
import core.utils.wah_utils as wah_utils
import ast

class WahSayPlan(SayPlan):
    def make_target_info(self,task_data, init_obs):
        target_info = {}
        target_info['nl_inst'] = task_data['nl_instructions'][0]
        target_info['init_obs_text'] = init_obs['obs_text']
        target_info['sayplan_dict'] = init_obs['sayplan_dict']
        return target_info
        
    def initial_logging(self, log, task_data, init_obs):
        log.info(f'Task ID: {task_data["task_id"]}')
        log.info(f'Your task is to: {task_data["nl_instructions"][0]}')
    
    def initial_logging_sem(self, log, task_data, sayplan_dict):
        log.info("Semantice Search Initialization")
        log.info(f'Your task is to: {task_data["nl_instructions"][0]}')
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)
        sg_custom = sg_json.replace('"','')
        log.info(f'3D Scene Graph: {sg_custom}')
        log.info('Memory: []')
    
    def initial_logging_iter(self, log, task_data, sayplan_dict, semantic_search_memory):
        log.info("Iterative Replanning Initialization")
        log.info(f'Your task is to: {task_data["nl_instructions"][0]}')
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)
        sg_custom = sg_json.replace('"','')
        log.info(f'3D Scene Graph: {sg_custom}')
        log.info(f'Memory: {semantic_search_memory}\n')


    def get_result(self, task_d):
        # task_goal, graph = task_d['task_goal'], self.env.get_graph()
        # name_id_dict_sim2nl, name_id_dict_nl2sim = self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim
        # subgoal_success_rate = wah_utils.check_goal_condition(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim)
        # if subgoal_success_rate == 1:
        #     goal_success_rate = 1
        # else:
        #     goal_success_rate = 0

        # result = {'task_id': task_d['task_id'],
        #             'nl_inst': task_d['nl_instructions'][0],
        #             'goal_success_rate': goal_success_rate,
        #             'subgoal_success_rate': subgoal_success_rate}
        # return result
        raise NotImplementedError
    
    def clean_sayplan_dict(self, sayplan_dict):
        cleaned = {"nodes": {}, "links": []}

        for category, nodes in sayplan_dict["nodes"].items():
            cleaned_nodes = []
            for node in nodes:
                new_node = {}
                for k, v in node.items():
                    # 빈 properties/states 제거
                    if k in ["properties", "states"] and isinstance(v, list) and not v:
                        continue
                    new_node[k] = v
                cleaned_nodes.append(new_node)
            cleaned["nodes"][category] = cleaned_nodes

        cleaned["links"] = sayplan_dict["links"]
        return cleaned

    def initial_collect_semantic_search(self, traj_file_path, task_data, sayplan_dict):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON 직렬화
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Key에서만 " 제거 (정규식: "key": → key:)
        sg_custom = sg_json.replace('"','')
        with open(traj_file_path, 'w') as f:
            f.write(f'Your task is to: {task_data["nl_instructions"][0]}\n')
            f.write(f"3D Scene Graph: {sg_custom}\n")
            f.write('Memory: []\n')

        print(f'Your task is to: {task_data["nl_instructions"][0]}\n')
        print(f"3D Scene Graph: {sg_custom}\n")
        print('Memory: []')
    
    def write_semantic_search(self, traj_file_path, sayplan_dict, memory):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON 직렬화
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Key에서만 " 제거 (정규식: "key": → key:)
        sg_custom = sg_json.replace('"','')
        with open(traj_file_path, 'a') as f:
            f.write(f"3D Scene Graph: {sg_custom}\n")
            f.write(f'Memory: {memory}\n')

        print(f"3D Scene Graph: {sg_custom}\n")
        print(f'Memory: {memory}')
        return sg_custom, memory
    
    def log_semantic_search(self, log, sayplan_dict, memory):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON 직렬화
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Key에서만 " 제거 (정규식: "key": → key:)
        sg_custom = sg_json.replace('"','')
        log.info(f"3D Scene Graph: {sg_custom}\n")
        log.info(f'Memory: {memory}\n')

        return sg_custom, memory

    def initial_collect_iterative_replanning(self, traj_file_path, task_data, sayplan_dict, semantic_search_memory):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON 직렬화
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Key에서만 " 제거 (정규식: "key": → key:)
        sg_custom = sg_json.replace('"','')
        with open(traj_file_path, 'w') as f:
            f.write(f'Your task is to: {task_data["nl_instructions"][0]}\n')
            f.write(f"3D Scene Graph: {sg_custom}\n")
            f.write(f'Memory: {semantic_search_memory}\n')
        
        print(f'Your task is to: {task_data["nl_instructions"][0]}\n')
        print(f"3D Scene Graph: {sg_custom}\n")
        print(f'Memory: {semantic_search_memory}')

    def write_text_traj(self, traj_file_path, text):
        with open(traj_file_path, 'a') as f:
            f.write(text + '\n')

    def simulate_plan(self, plan, task_data, skill_set=None):
        self.env.reset(task_data)
        steps = [action.strip() for action in plan.strip("[]").split(",")]

        if skill_set is None:
            for next_step in steps:
                obs = self.env.step(next_step)
                if not obs['success']:
                    feedback = obs['feedback']
                    return feedback
        else:
            for next_step in steps:
                if next_step not in skill_set:
                    feedback = f"Invalid action: {next_step}."
                    return feedback
                else:
                    obs = self.env.step(next_step)
                    if not obs['success']:
                        feedback = obs['feedback']
                        return feedback
        return 'done'
    
    def get_semantic_skill_set(self, scene_graph):
        room_nodes = [n for n, attr in scene_graph.nodes(data=True) if attr.get('node_type') == 'room']
        skill_set = []
        for room in room_nodes:
            skill_set.append(f'expand({room})')
            skill_set.append(f'contract({room})')
        skill_set.append('done')
        return skill_set

    def get_iterative_skill_set(self, env):
        sim_graph = env.get_graph()
        filtered_sim_objs = env.filtered_sim_objs
        name_id_dict_sim2nl = env.name_id_dict_sim2nl
        agent_id = env.agent_id

        filtered_graph = wah_utils.extract_graph_by_class_names(sim_graph, filtered_sim_objs)
        obs_all_rooms = wah_utils.obs_all_rooms(filtered_graph)
        obs_partial_objs = wah_utils.obs_partial_objs(filtered_graph, agent_id)

        nl_obs_all_rooms_info = [name_id_dict_sim2nl[obs_room] for obs_room in obs_all_rooms]
        nl_obs_partial_objs_info = [name_id_dict_sim2nl[partial_obj] for partial_obj in obs_partial_objs]

        nl_grabbable_names = ['alcohol', 'apple', 'bananas', 'bar soap', 'bell pepper', 'board game', 'book', 'box', 'slice of bread', 'bucket', 'candle', 'candy bar', 'carrot', 'cell phone', 'cereal', 'chair', 'chicken', 'Chinese food', 'chips', 'chocolate syrup', 'clock', 'pants', 'pile of clothes', 'shirt', 'coat rack', 'coffee pot', 'condiment bottle', 'condiment shaker', 'cooking pot', 'crackers', 'crayons', 'creamy buns', 'cupcake', 'cutlery fork', 'cutlery knife', 'cutlets', 'cutting board', 'bowl', 'dishwashing liquid', 'face cream', 'folder', 'frying pan', 'glasses', 'globe', 'hair product', 'hanger', 'juice', 'keyboard', 'lime', 'lotion bottle', 'magazine', 'milk', 'milkshake', 'minced meat', 'mouse', 'mug', 'notes', 'oven tray', 'pancake', 'paper', 'pear', 'pie', 'pillow', 'plate', 'plum', 'pound cake', 'pudding', 'radio', 'remote control', 'rug', 'salad', 'salmon', 'slippers', 'sports ball', 'sundae', 'teddy bear', 'toilet paper', 'toothbrush', 'toothpaste', 'towel', 'towel rack', 'toy', 'wall phone', 'wall picture frame', 'washing sponge', 'water glass', 'whipped cream', 'wine', 'wine glass']
        nl_open_names = ['bathroom cabinet', 'book', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee pot', 'cooking pot', 'curtains', 'desk', 'dishwasher', 'door', 'folder', 'fridge', 'garbage can', 'hair product', 'kitchen cabinet', 'lotion bottle', 'magazine', 'microwave oven', 'milk', 'nightstand', 'printer', 'radio', 'stove', 'toilet', 'toothpaste', 'washing machine', 'window']
        nl_switch_names = ['candle', 'cell phone', 'clock', 'computer', 'dishwasher', 'faucet', 'fridge', 'light switch', 'microwave oven', 'printer', 'radio', 'remote control', 'stove', 'toaster', 'tv', 'wall phone', 'washing machine']

        skill_set = ['done', 'failure']
        skill_set += [f'go to {room_info[0]} {room_info[1]}' for room_info in nl_obs_all_rooms_info]
        for partial_obj_info in nl_obs_partial_objs_info:
            nl_obj_name, nl_obj_id = partial_obj_info[0], partial_obj_info[1]
            skill_set.append(f'go to {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_grabbable_names:
                skill_set.append(f'pick up {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'put down {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_open_names:
                skill_set.append(f'open {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'close {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_switch_names:
                skill_set.append(f'turn on {nl_obj_name} {nl_obj_id}')
        return skill_set