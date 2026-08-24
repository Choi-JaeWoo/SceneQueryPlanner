import os
import sys
import json
import copy
from typing import Dict, Any


from core.env.base_env import BaseEnv
from virtualhome.simulation.environment.unity_environment import UnityEnvironment
import core.utils.wah_utils as wah_utils



class WahEnv(BaseEnv):
    def __init__(self, cfg):
        self.cfg = cfg

        self.env = UnityEnvironment(
            num_agents=1,
            observation_types=self.cfg.environment.observation_types,
            use_editor=self.cfg.environment.use_editor,
            base_port=self.cfg.environment.base_port,
            port_id=self.cfg.environment.port_id,
            executable_args=self.cfg.environment.executable_args,
            recording_options=self.cfg.environment.recording_options
        )

        self.comm = self.env.comm
        self.agent_id = 1
        self.agent_reset_id = self.cfg.environment.agent_reset_id
        self.recording_options = self.cfg.environment.recording_options
        self.default_image_width = 300
        self.default_image_height = 300

        # object name dicts
        with open(cfg.dataset.obj_dict_sim2nl, 'r') as f:
            self.obj_dict_sim2nl = json.load(f)
        with open(cfg.dataset.obj_dict_nl2sim, 'r') as f:
            self.obj_dict_nl2sim = json.load(f)

        self.max_ids = self.env.max_ids
        self.filtered_sim_objs = cfg.environment.filtered_sim_objs
        self.sim_receps = cfg.environment.sim_receps
        self.cur_recep_info = (None, None)
        print("[WAH] WahEnv initialized.")

    def reset(self, task_d):
        self.task_id = task_d['task_id']
        self.init_graph = copy.deepcopy(task_d['init_graph'])
        self.init_room = task_d['init_room']
        self.task_goal = task_d['task_goal']
        self.task_name = task_d['task_name']
        self.env_id = task_d['env_id']
        
        self.comm.reset(self.env_id)
        _, g = self.comm.environment_graph()
        edge_ids = {e['to_id'] for e in g['edges']} | {e['from_id'] for e in g['edges']}
        node_ids = {n['id'] for n in g['nodes']}
        assert edge_ids <= node_ids, "Graph error: edges refer to nonexistent nodes"
        
        if self.env_id not in self.max_ids:
            self.max_ids[self.env_id] = max(n['id'] for n in g['nodes'])

        updated_graph = wah_utils.separate_new_ids_graph(self.init_graph, self.max_ids[self.env_id])
        success, msg = self.comm.expand_scene(updated_graph)
        if not success:
            raise RuntimeError(f"[WAH] Expand scene failed: {msg}")
        
        self.offset_cameras = self.comm.camera_count()[1]
        self.comm.add_character(self.env.agent_info[self.agent_reset_id], initial_room=self.init_room)
        self.changed_graph = True
        # _, self.init_unity_graph = self.comm.environment_graph()
        self.init_unity_graph = self.get_graph()

        self.id2node = {node['id']: node for node in self.init_unity_graph['nodes']}
        self.name_id_dict_sim2nl, self.name_id_dict_nl2sim = wah_utils.make_name_id_dict(self.init_unity_graph, self.obj_dict_sim2nl)
        self.vis_log = [{'action': 'init', 'images': self.get_visual_obs()}]
        self.working_memory = {}

        obs_text = self.get_text_obs()
        obs_vis = self.get_visual_obs()

        results = {'obs_text': obs_text,
                   'obs_vis': obs_vis,}
        return results


    def step(self, nl_action):
        # Check script executability
        possible, feedback = self.get_text_feedback(nl_action)

        # Make script    
        sim_skill_info = wah_utils.decompose_nl_skill(nl_action, self.name_id_dict_nl2sim, self.cur_recep_info)
        script = wah_utils.make_script(sim_skill_info['sim_act'], sim_skill_info['sim_obj_info'], self.cur_recep_info)
        
        # Try to execute script
        if possible:
            script_list = [script]
            if self.recording_options['recording']:
                success, message = self.comm.render_script(script_list,
                                                        find_solution=False,
                                                        processing_time_limit=60,
                                                        recording=True,
                                                        skip_animation=False,
                                                        camera_mode=list(self.recording_options['cameras']),
                                                        output_folder=self.recording_options['output_folder'],
                                                        file_name_prefix=self.recording_options['file_name_prefix'])
            else:
                success, message = self.comm.render_script(script_list,
                                                        find_solution=False,
                                                        recording=False,
                                                        skip_animation=True)
            if success:
                self.changed_graph = True
                vis_log_text = nl_action
            # Execution failure -> low-level controller problem
            # TODO: Check if the failure is due to low-level controller problem
            else:
                sim_act, sim_obj_info = sim_skill_info['sim_act'], sim_skill_info['sim_obj_info']
                nl_obj_info = self.name_id_dict_sim2nl[sim_obj_info]
                if sim_act == 'walk':
                    feedback = f"You can't go to {nl_obj_info[0]} ({nl_obj_info[1]}) because of low-level controller problem."
                elif sim_act == 'grab':
                    feedback = f"You can't pick up {nl_obj_info[0]} ({nl_obj_info[1]}) because of low-level controller problem."
                elif sim_act == 'putin':
                    sim_recep_info = sim_skill_info['sim_recep_info']
                    nl_recep_info = self.name_id_dict_sim2nl[sim_recep_info]
                    feedback = f"You can't put {nl_obj_info[0]} ({nl_obj_info[1]}) in {nl_recep_info[0]} ({nl_recep_info[1]}) because of low-level controller problem."
                elif sim_act == 'putback':
                    sim_recep_info = sim_skill_info['sim_recep_info']
                    nl_recep_info = self.name_id_dict_sim2nl[sim_recep_info]
                    feedback = f"You can't put {nl_obj_info[0]} ({nl_obj_info[1]}) in {nl_recep_info[0]} ({nl_recep_info[1]}) because of low-level controller problem."
                elif sim_act == 'open':
                    feedback = f"You can't open {nl_obj_info[0]} ({nl_obj_info[1]}) because of low-level controller problem."
                elif sim_act == 'close':
                    feedback = f"You can't close {nl_obj_info[0]} ({nl_obj_info[1]}) because of low-level controller problem."
                elif sim_act == 'switchon':
                    feedback = f"You can't turn on {nl_obj_info[0]} ({nl_obj_info[1]}) because of low-level controller problem."
                self.changed_graph = False
                vis_log_text = f'{nl_action} (execution fail)'
        else:
            success = False
            self.changed_graph = False
            vis_log_text = f'{nl_action} (fail)'
        
        # If success, update the current receptacle info & get text observation
        if success:
            if (sim_skill_info['sim_act']) == 'walk' and (sim_skill_info['sim_obj_info'][0] in self.sim_receps):
                self.cur_recep_info = sim_skill_info['sim_obj_info']
            obs_text = self.get_text_obs(nl_action)
        else:
            obs_text = None
        
        obs_vis = self.get_visual_obs()
        self.vis_log.append({'action': vis_log_text, 'images': obs_vis})
        
        results = {'success': success,
                   'feedback': feedback,
                   'obs_text': obs_text,
                   'obs_vis': obs_vis,
                   'reason_type': 'controller' if possible and not success else ('precondition' if not possible else None),
                   }
        
        return results


    def get_visual_obs(self, camera_info={}):
        camera_ids = [self.offset_cameras + 3] # [self.offset_cameras + 1, self.offset_cameras + 2, self.offset_cameras + 3, self.offset_cameras + 4, self.offset_cameras + 5, self.offset_cameras + 6]
        width = camera_info.get('image_width', self.default_image_width)
        height = camera_info.get('image_height', self.default_image_height)
        mode = camera_info.get('mode', 'normal')

        success, images = self.comm.camera_image(camera_ids, mode=mode, image_width=width, image_height=height)
        if not success:
            raise RuntimeError("[WAH] Camera image failed.")
        return images
    
    def get_text_obs(self, nl_action=None):
        full_graph = self.get_graph()
        partial_graph = self.get_partial_graph()
        agent_id = self.agent_id
        if nl_action == None:
            obs_all_rooms = wah_utils.obs_all_rooms(partial_graph)
            obs_agent_room = wah_utils.obs_agent_room(partial_graph, agent_id)
            obs_room_items = wah_utils.obs_room_items(partial_graph)
            obs_agent_room_nl = self.name_id_dict_sim2nl[obs_agent_room]
            obs_text = f'You are in the house, and there are {len(obs_all_rooms)} rooms: {wah_utils.merge_obs_list(obs_all_rooms, self.name_id_dict_sim2nl)}. '
            obs_text += f'You are in the middle of a {obs_agent_room_nl[0]} {obs_agent_room_nl[1]}. '
            obs_text += f'Looking quickly around the room, you see {wah_utils.merge_obs_list(obs_room_items, self.name_id_dict_sim2nl)}.'
        else:
            sim_skill_info = wah_utils.decompose_nl_skill(nl_action, self.name_id_dict_nl2sim, self.cur_recep_info)
            sim_act, sim_obj_info = sim_skill_info['sim_act'], sim_skill_info['sim_obj_info']
            nl_obj_info = self.name_id_dict_sim2nl[sim_obj_info]

            if sim_act == 'walk':
                if sim_obj_info[0] in ['bathroom', 'bedroom', 'kitchen', 'livingroom']:
                    obs_room_items = wah_utils.obs_room_items(partial_graph)
                    obs_text = f'You move to the {nl_obj_info[0]} {nl_obj_info[1]}. '
                    obs_text += f'Looking quickly around the room, you see {wah_utils.merge_obs_list(obs_room_items, self.name_id_dict_sim2nl)}.'
                else:
                    obs_text = f'You arrive at the {nl_obj_info[0]} {nl_obj_info[1]}. '
                    if wah_utils.check_properties(full_graph, sim_obj_info[1], 'CAN_OPEN') and not(nl_obj_info[0] == 'desk'):
                        if wah_utils.check_states(full_graph, sim_obj_info[1], 'CLOSED'):
                            obs_text += f'The {nl_obj_info[0]} {nl_obj_info[1]} is closed. '
                        elif wah_utils.check_states(full_graph, sim_obj_info[1], 'OPEN'):
                            obs_text += f'The {nl_obj_info[0]} {nl_obj_info[1]} is open. '
                    obs_close_objs = wah_utils.obs_close_objs(partial_graph, agent_id)
                    obs_text += f'You see {wah_utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}.'
            elif sim_act == 'grab':
                obs_text = f'You pick up {nl_obj_info[0]} {nl_obj_info[1]}.'
            elif sim_act == 'putin':
                sim_recep_info = sim_skill_info['sim_recep_info']
                nl_recep_info = self.name_id_dict_sim2nl[sim_recep_info]
                obs_text = f"You put {nl_obj_info[0]} {nl_obj_info[1]} in {nl_recep_info[0]} {nl_recep_info[1]}."
            elif sim_act == 'putback':
                sim_recep_info = sim_skill_info['sim_recep_info']
                nl_recep_info = self.name_id_dict_sim2nl[sim_recep_info]
                obs_text = f"You put {nl_obj_info[0]} {nl_obj_info[1]} on {nl_recep_info[0]} {nl_recep_info[1]}."
            elif sim_act == 'open':
                obs_text = f'You open {nl_obj_info[0]} {nl_obj_info[1]}. '
                obs_close_objs = wah_utils.obs_close_objs(partial_graph, agent_id)
                obs_text += f'You see {wah_utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}'
            elif sim_act == 'close':
                obs_text = f'You close {nl_obj_info[0]} {nl_obj_info[1]}.'
            elif sim_act == 'switchon':
                obs_text = f'You turn on {nl_obj_info[0]} {nl_obj_info[1]}.'
        obs_grab_objs = wah_utils.obs_agent_grab(partial_graph, agent_id)
        if not obs_grab_objs == None:
            obs_text += f' You hold {wah_utils.merge_obs_list(obs_grab_objs, self.name_id_dict_sim2nl)}.'
        return obs_text

    def get_graph_obs(self, visibility='full'):
        if visibility == 'full':
            return self.get_graph()
        elif visibility == 'partial':
            return self.get_partial_graph()
        elif visibility == 'initial':
            full_graph = self.get_graph()
            filtered_sim_objs = self.filtered_sim_objs        
            filtered_graph = wah_utils.extract_graph_by_class_names(full_graph, filtered_sim_objs)
            return filtered_graph
        else:
            raise NotImplementedError(f"Visibility type '{visibility}' is not implemented.")

    def get_skill_set(self):
        partial_graph = self.get_partial_graph()
        obs_all_rooms = wah_utils.obs_all_rooms(partial_graph)
        obs_partial_objs = wah_utils.obs_partial_objs(partial_graph, self.agent_id) 
        nl_obs_all_rooms_info = [self.name_id_dict_sim2nl[obs_room] for obs_room in obs_all_rooms]
        nl_obs_partial_objs_info = [self.name_id_dict_sim2nl[partial_obj] for partial_obj in obs_partial_objs]

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
        # if self.is_working_memory:
        #     for nl_grabbalbe_obj in nl_grabbable_names:
        #         skill_set.append(f'recall location of {nl_grabbalbe_obj}')
        # if self.is_scene_graph:
        #     skill_set.append(f'retrieve information of {nl_grabbalbe_obj}')
        return skill_set

    def get_text_feedback(self, nl_action=None):
        sim_skill_info = wah_utils.decompose_nl_skill(nl_action, self.name_id_dict_nl2sim, self.cur_recep_info)
        sim_act, sim_obj_info = sim_skill_info['sim_act'], sim_skill_info['sim_obj_info']
        graph = self.get_graph()
        name_id_dict_sim2nl = self.name_id_dict_sim2nl
        agent_id = self.agent_id
        nl_obj_info = name_id_dict_sim2nl[sim_obj_info]
        if sim_act == 'walk':
            possible = True
            feedback = None
        elif sim_act == 'grab':
            ### Check: 1) free hand 2) obj_close 3) obj_in_open_recep 4) obj_grabbable
            is_free_hand = wah_utils.check_free_hand(graph, agent_id)
            is_obj_close = wah_utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_in_open_recep, closed_recep_info = wah_utils.check_obj_in_open_recep(graph, sim_obj_info[1])
            is_obj_grabbable = wah_utils.check_properties(graph, sim_obj_info[1], 'GRABBABLE')
            if not closed_recep_info == None:
                nl_closed_recep_info = name_id_dict_sim2nl[closed_recep_info]
            
            if not is_free_hand:
                feedback = f"You can't pick up {nl_obj_info[0]} {nl_obj_info[1]} because you don't have an empty hand."
            elif not is_obj_close:
                feedback = f"You can't pick up {nl_obj_info[0]} {nl_obj_info[1]} because it's not close to you."
            elif not is_obj_in_open_recep:
                feedback = f"You can't pick up {nl_obj_info[0]} {nl_obj_info[1]} because it's inside closed {nl_closed_recep_info[0]} ({nl_closed_recep_info[1]})."
            elif not is_obj_grabbable:
                feedback = f"You can't pick up {nl_obj_info[0]} {nl_obj_info[1]} because it can't be grabbed."
            else:
                feedback = None
            possible = is_free_hand and is_obj_close and is_obj_in_open_recep and is_obj_grabbable
        elif sim_act == 'putin':
            sim_recep_info = sim_skill_info['sim_recep_info']
            nl_recep_info = name_id_dict_sim2nl[sim_recep_info]
            is_holding_obj = wah_utils.check_holding_obj(graph, agent_id, sim_obj_info[1])
            is_recep_close = wah_utils.check_obj_close_to_agent(graph, agent_id, sim_recep_info[1])
            is_recep_container = wah_utils.check_properties(graph, sim_recep_info[1], 'CONTAINERS')
            is_not_recep_closed = not wah_utils.check_states(graph, sim_recep_info[1], 'CLOSED')
            if not is_holding_obj:
                feedback = f"You can't put {nl_obj_info[0]} {nl_obj_info[1]} in {nl_recep_info[0]} {nl_recep_info[1]} because you're not holding it."
            elif not is_recep_close:
                feedback = f"You can't put {nl_obj_info[0]} {nl_obj_info[1]} in {nl_recep_info[0]} {nl_recep_info[1]} because the {nl_recep_info[0]} is not close to you."
            elif not is_recep_container:
                feedback = f"You can't put {nl_obj_info[0]} {nl_obj_info[1]} in {nl_recep_info[0]} {nl_recep_info[1]} because the {nl_recep_info[0]} is not a container."
            elif not is_not_recep_closed:
                feedback = f"You can't put {nl_obj_info[0]} {nl_obj_info[1]} in {nl_recep_info[0]} {nl_recep_info[1]} because the {nl_recep_info[0]} is closed."
            else:
                feedback = None
            possible = is_holding_obj and is_recep_close and is_recep_container and is_not_recep_closed
        elif sim_act == 'putback':
            ### Check: 1) agent holding obj 2) recep close 3) recep surface
            sim_recep_info = sim_skill_info['sim_recep_info']
            nl_recep_info = name_id_dict_sim2nl[sim_recep_info]
            is_holding_obj = wah_utils.check_holding_obj(graph, agent_id, sim_obj_info[1])
            is_recep_close = wah_utils.check_obj_close_to_agent(graph, agent_id, sim_recep_info[1])
            is_recep_surface = wah_utils.check_properties(graph, sim_recep_info[1], 'SURFACES')
            if not is_holding_obj:
                feedback = f"You can't put {nl_obj_info[0]} {nl_obj_info[1]} on {nl_recep_info[0]} {nl_recep_info[1]} because you're not holding it."
            elif not is_recep_close:
                feedback = f"You can't put {nl_obj_info[0]} {nl_obj_info[1]} on {nl_recep_info[0]} {nl_recep_info[1]} because the {nl_recep_info[0]} is not close to you."
            elif not is_recep_surface:
                feedback = f"You can't put {nl_obj_info[0]} {nl_obj_info[1]} in {nl_recep_info[0]} {nl_recep_info[1]} because the {nl_recep_info[0]} is not a surface."
            else:
                feedback = None
            possible = is_holding_obj and is_recep_close and is_recep_surface
        elif sim_act == 'open':
            ### Check: 1) free hand 2) obj close 3) obj opennable 4) obj closed
            is_free_hand = wah_utils.check_free_hand(graph, agent_id)
            is_obj_close = wah_utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_opennable = wah_utils.check_properties(graph, sim_obj_info[1], 'CAN_OPEN')
            is_obj_closed = wah_utils.check_states(graph, sim_obj_info[1], 'CLOSED')
            if not is_free_hand:
                feedback = f"You can't open {nl_obj_info[0]} {nl_obj_info[1]} because you don't have an empty hand."
            elif not is_obj_close:
                feedback = f"You can't open {nl_obj_info[0]} {nl_obj_info[1]} because it's not close to you."
            elif not is_obj_opennable:
                feedback = f"You can't open {nl_obj_info[0]} {nl_obj_info[1]} because it can't be opened."
            elif not is_obj_closed:
                feedback = f"You can't open {nl_obj_info[0]} {nl_obj_info[1]} because it's already open."
            else:
                feedback = None
            possible = is_free_hand and is_obj_close and is_obj_opennable and is_obj_closed
        elif sim_act == 'close':
            ### Check: 1) free hand 2) obj close 3) obj opennable 4) obj open
            is_free_hand = wah_utils.check_free_hand(graph, agent_id)
            is_obj_close = wah_utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_opennable = wah_utils.check_properties(graph, sim_obj_info[1], 'CAN_OPEN')
            is_obj_open = wah_utils.check_states(graph, sim_obj_info[1], 'OPEN')
            if not is_free_hand:
                feedback = f"You can't close {nl_obj_info[0]} {nl_obj_info[1]} because you don't have an empty hand."
            elif not is_obj_close:
                feedback = f"You can't close {nl_obj_info[0]} {nl_obj_info[1]} because it's not close to you."
            elif not is_obj_opennable:
                feedback = f"You can't close {nl_obj_info[0]} {nl_obj_info[1]} because it can't be closed."
            elif not is_obj_open:
                feedback = f"You can't close {nl_obj_info[0]} {nl_obj_info[1]} because it's already closed."
            else:
                feedback = None
            possible = is_free_hand and is_obj_close and is_obj_opennable and is_obj_open
        elif sim_act == 'switchon':
            ### Check: 1) free hand 2) obj close 3) obj has_switch 4) obj off
            is_free_hand = wah_utils.check_free_hand(graph, agent_id)
            is_obj_close = wah_utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_hasswitch = wah_utils.check_properties(graph, sim_obj_info[1], 'HAS_SWITCH')
            is_obj_off = wah_utils.check_states(graph, sim_obj_info[1], 'OFF')
            if not is_free_hand:
                feedback = f"You can't turn on {nl_obj_info[0]} {nl_obj_info[1]} because you don't have an empty hand."
            elif not is_obj_close:
                feedback = f"You can't turn on {nl_obj_info[0]} {nl_obj_info[1]} because it's not close to you."
            elif not is_obj_hasswitch:
                feedback = f"You can't turn on {nl_obj_info[0]} {nl_obj_info[1]} because it doesn't have a switch."
            elif not is_obj_off:
                feedback = f"You can't turn on {nl_obj_info[0]} {nl_obj_info[1]} because it's already turned on."
            else:
                feedback = None
            possible = is_free_hand and is_obj_close and is_obj_hasswitch and is_obj_off
        else:
            raise NotImplementedError()
        return possible, feedback

    def get_graph(self):
        if self.changed_graph:
            s, graph = self.comm.environment_graph()
            if not s:
                raise RuntimeError(f'Failed to fetch environment graph: {graph}')
            self.graph = graph
            self.changed_graph = False
        return self.graph
    
    def get_partial_graph(self):
        full_graph = self.get_graph()
        filtered_sim_objs = self.filtered_sim_objs        
        filtered_graph = wah_utils.extract_graph_by_class_names(full_graph, filtered_sim_objs)
        agent_id = self.agent_id
        partial_graph = wah_utils.get_visible_nodes(filtered_graph, agent_id)
        return partial_graph

