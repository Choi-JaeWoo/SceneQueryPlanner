import os
import sys
import json
import re


from core.planner.procthor.sayplan import SayPlan
import core.utils.procthor_utils as procthor_utils
import ast

class ProcThorSayPlan(SayPlan):
    def make_target_info(self,task_data, init_obs):
        target_info = {}
        target_info['nl_inst'] = task_data['instruction'][0]
        target_info['init_obs_text'] = init_obs['obs_text']
        target_info['sayplan_dict'] = init_obs['sayplan_dict']
        return target_info
    
    def initial_logging(self, log, task_data, init_obs):
        log.info(f'Task ID: {task_data["env_id"]}')
        log.info(f'Your task is to: {task_data["instruction"][0]}')
    
    def initial_logging_sem(self, log, task_data, sayplan_dict):
        log.info("Semantice Search Initialization")
        log.info(f'Your task is to: {task_data["instruction"][0]}')
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)
        sg_custom = sg_json.replace('"','')
        log.info(f'3D Scene Graph: {sg_custom}')
        log.info('Memory: []')
    
    def initial_logging_iter(self, log, task_data, sayplan_dict, semantic_search_memory):
        log.info("Iterative Replanning Initialization")
        log.info(f'Your task is to: {task_data["instruction"][0]}')
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)
        sg_custom = sg_json.replace('"','')
        log.info(f'3D Scene Graph: {sg_custom}')
        log.info(f'Memory: {semantic_search_memory}\n')

    def get_result(self, task_d):
        raise NotImplementedError
    
    def clean_sayplan_dict(self, sayplan_dict):
        cleaned = {"nodes": {}, "links": []}
        properties_keys = [
            'toggleable', 'canFillWithLiquid', 'cookable', 'breakable',
            'isHeatSource', 'isColdSource', 'sliceable', 
            'openable', 'pickupable'
        ]
        states_keys = [
            'isToggled', 'isFilledWithLiquid', 'isCooked',
            'isSliced', 'isOpen', 'isPickedUp', 'isBroken'
        ]
        for category, nodes in sayplan_dict["nodes"].items():
            cleaned_nodes = []
            for node in nodes:
                new_node = {}
                for k, v in node.items():
                    if k in properties_keys or k in states_keys:
                        continue  
                    new_node[k] = v

                properties = [key for key in properties_keys if node.get(key, False) is True]
                states = [key for key in states_keys if node.get(key, False) is True]

                if properties:
                    new_node["properties"] = properties
                if states:
                    new_node["states"] = states

                cleaned_nodes.append(new_node)
            cleaned["nodes"][category] = cleaned_nodes
        cleaned["links"] = sayplan_dict["links"]
        return cleaned

    def initial_collect_semantic_search(self, traj_file_path, task_data, sayplan_dict):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON serialization
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Remove " only from keys (regex: "key": -> key:)
        sg_custom = sg_json.replace('"','')
        with open(traj_file_path, 'w') as f:
            f.write(f'Your task is to: {task_data["instruction"][0]}\n')
            f.write(f"3D Scene Graph: {sg_custom}\n")
            f.write('Memory: []\n')

        print(f'Your task is to: {task_data["instruction"][0]}\n')
        print(f"3D Scene Graph: {sg_custom}\n")
        print('Memory: []')
    
    def write_semantic_search(self, traj_file_path, sayplan_dict, memory):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON serialization
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Remove " only from keys (regex: "key": -> key:)
        sg_custom = sg_json.replace('"','')
        with open(traj_file_path, 'a') as f:
            f.write(f"3D Scene Graph: {sg_custom}\n")
            f.write(f'Memory: {memory}\n')

        print(f"3D Scene Graph: {sg_custom}\n")
        print(f'Memory: {memory}')
        return sg_custom, memory
    
    def log_semantic_search(self, log, sayplan_dict, memory):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON serialization
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Remove " only from keys (regex: "key": -> key:)
        sg_custom = sg_json.replace('"','')
        log.info(f"3D Scene Graph: {sg_custom}\n")
        log.info(f'Memory: {memory}\n')

        return sg_custom, memory
    
    def initial_collect_iterative_replanning(self, traj_file_path, task_data, sayplan_dict, semantic_search_memory):
        cleaned_sg = self.clean_sayplan_dict(sayplan_dict)
        # JSON serialization
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)

        # Remove " only from keys (regex: "key": -> key:)
        sg_custom = sg_json.replace('"','')
        with open(traj_file_path, 'w') as f:
            f.write(f'Your task is to: {task_data["instruction"][0]}\n')
            f.write(f"3D Scene Graph: {sg_custom}\n")
            f.write(f'Memory: {semantic_search_memory}\n')
        
        print(f'Your task is to: {task_data["instruction"][0]}\n')
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
                skill_set = self.get_iterative_skill_set(self.env)
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
        if sim_graph == None:
            return ['done', 'failure']
        name_id_dict_sim2nl = env.name_id_dict_sim2nl
        
        obs_all_rooms = []
        for rid, info in env.room_info.items():
            obs_all_rooms.append((info["roomType"], rid))
        obs_partial_objs = procthor_utils.obs_partial_objs(sim_graph)
        
        nl_obs_all_rooms_info = [
            name_id_dict_sim2nl[obs_room]
            for obs_room in obs_all_rooms
            if obs_room in name_id_dict_sim2nl
        ]
        nl_obs_partial_objs_info = [
            name_id_dict_sim2nl[partial_obj]
            for partial_obj in obs_partial_objs
            if partial_obj in name_id_dict_sim2nl
        ]
        nl_grabbable_names = ['alarm clock', 'aluminum foil', 'apple', 'sliced apple', 'baseball bat', 'basket ball', 'book', 'boots', 'bottle', 'bowl', 'box', 'bread', 'sliced bread', 'butter knife', 'candle', 'CD', 'cell phone', 'cloth', 'credit card', 'cup', 'dish sponge', 'dumbbell', 'egg', 'cracked egg', 'fork', 'hand towel', 'kettle', 'key chain', 'knife', 'ladle', 'laptop', 'lettuce', 'sliced lettuce', 'mug', 'newspaper', 'pan', 'paper towel roll', 'pen', 'pencil', 'pepper shaker', 'pillow', 'plate', 'plunger', 'pot', 'potato', 'sliced potato', 'remote control', 'salt shaker', 'scrub brush', 'soap bar', 'soap bottle', 'spatula', 'spoon', 'spray bottle', 'statue', 'tabletop decor', 'teddy bear', 'tennis racket', 'tissue box', 'toilet paper', 'tomato', 'sliced tomato', 'towel', 'vase', 'watch', 'watering can', 'wine bottle']
        nl_open_names = ['blinds', 'book', 'box', 'cabinet', 'drawer', 'fridge', 'kettle', 'laptop', 'microwave', 'safe', 'shower curtain', 'shower door', 'toilet']
        nl_switch_names = ['candle', 'cell phone', 'coffee machine', 'desk lamp', 'faucet', 'floor lamp', 'laptop', 'light switch', 'microwave', 'shower head', 'stove burner', 'stove knob', 'television', 'toaster']
        nl_slice_names = ['apple', 'bread', 'lettuce', 'potato', 'tomato']
        
        skill_set = ['done', 'failure']
        skill_set += [f'go to {room_info[0]} {room_info[1]}' for room_info in nl_obs_all_rooms_info]
        for partial_obj_info in nl_obs_partial_objs_info:
            nl_obj_name, nl_obj_id = partial_obj_info[0], partial_obj_info[1]
            skill_set.append(f'go to {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_grabbable_names:
                skill_set.append(f'pick up {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'put down {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'drop {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_open_names:
                skill_set.append(f'open {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'close {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_switch_names:
                skill_set.append(f'turn on {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'turn off {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_slice_names:
                skill_set.append(f'slice {nl_obj_name} {nl_obj_id}')
        return skill_set