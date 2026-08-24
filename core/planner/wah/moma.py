import os
import sys
from collections import defaultdict, Counter


from core.planner.base_planner import BasePlanner
from core.wm.sg_wah import WahSG_One
# from wm.sg_procthor import ProcThorSG
# from wm.sg_procthor import ProcThorSG_One
# from wm.sg_procthor import draw_scene_graph




class MoMa(BasePlanner):
    def __init__(self, cfg, env, llm_agent, retriever):
        super().__init__(cfg, env, llm_agent)
        self.max_decision_step = cfg.llm_agent.max_decision_step
        self.cur_decision_step = cfg.llm_agent.initial_step
        self.retriever = retriever
        self.history = []

    def run(self, task_data, log):
        self.cur_decision_step = 0
        self.history = []

        init_obs = self.env.reset(task_data)
        
        partial_graph = self.env.get_graph_obs(visibility='initial')
        working_memory = WahSG_One(self.cfg, partial_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim)
        
        target_info = self.make_target_info(task_data, init_obs)
        
        self.initial_logging(log, task_data)

        observation = init_obs['obs_text']
        previous_step = None
        
        while True:
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                log.info('Max decision steps reached.')
                return terminate_info
            
            skill_set = self.env.get_skill_set()

            try:
                prompt_obs = self.make_observation(observation, working_memory.scene_graph, previous_step)
                log.info(prompt_obs)
                self.llm_agent.reset(target_info, self.history)
                self.llm_agent.add_text_traj(prompt_obs)
                next_step_info = self.llm_agent.decision_making(skill_set)
                next_step_class, next_step, reasoning = next_step_info['next_step_class'], next_step_info['next_step'], next_step_info['reasoning']
                log.info(f'Think: {reasoning}')
                log.info(f'{next_step_class}: {next_step}')
                
            except Exception as e:
                terminate_info = {'terminate': 'plan_next_step_error', 'decision_step': self.cur_decision_step}
                log.info(f"Plan Next Step Error: {e}")
                return terminate_info

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
                    working_memory.update_nx_graph(sim_graph, next_step.split('go to ')[-1])    
                else:
                    working_memory.update_nx_graph(sim_graph)
                if obs['success']:
                    observation = obs['obs_text']
                    previous_step = (next_step, obs['success'])
                else:
                    observation = obs['feedback']
                    previous_step = (next_step, obs['success'])
                self.cur_decision_step += 2


    def collect_human(self, task_data, collect_dir):
        self.cur_decision_step = 0
        self.history = []
        
        init_obs = self.env.reset(task_data)

        partial_graph = self.env.get_graph_obs(visibility='initial')
        working_memory = WahSG_One(self.cfg, partial_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim)
        
        observation = init_obs['obs_text']
        previous_step = None

        while True:
            traj_file_path = os.path.join(collect_dir, f"traj_{task_data['task_id']:03d}_{self.cur_decision_step:03d}.txt")
            self.initial_collect(traj_file_path, task_data)
            
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                with open(traj_file_path, 'a') as f:
                    f.write('Max decision steps reached.')
                return terminate_info
            
            skill_set = self.env.get_skill_set()
            
            prompt_obs = self.make_observation(observation, working_memory.scene_graph, previous_step)
            print(prompt_obs)
            with open(traj_file_path, 'a') as f:
                f.write(f'{prompt_obs}')
            
            next_step = input('Think: ')
            self.cur_decision_step += 1
            with open(traj_file_path, 'a') as f:
                f.write(f'Think: {next_step}\n')

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
                        working_memory.update_nx_graph(sim_graph, next_step.split('go to ')[-1])    
                    else:
                        working_memory.update_nx_graph(sim_graph)
                    if obs['success']:
                        observation = obs['obs_text']
                        previous_step = (next_step, obs['success'])
                    else:
                        observation = obs['feedback']
                        previous_step = (next_step, obs['failure'])
                    self.cur_decision_step +=1
                    print(observation)
            else:
                print(f"Invalid action: {next_step}. Please choose from {skill_set}.")
            
    def collect_llm(self, task_data, collect_dir):
        self.cur_decision_step = 0
        self.history = []

        init_obs = self.env.reset(task_data)
        
        partial_graph = self.env.get_graph_obs(visibility='initial')
        working_memory = WahSG_One(self.cfg, partial_graph, self.env.name_id_dict_sim2nl, self.env.name_id_dict_nl2sim)
        
        target_info = self.make_target_info(task_data, init_obs)
        
        observation = init_obs['obs_text']
        previous_step = None
        
        

        while True:
            traj_file_path = os.path.join(collect_dir, f"traj_{task_data['task_id']:03d}_{self.cur_decision_step:03d}.txt")
            # traj_file_path = os.path.join(collect_dir, f"traj_{task_data['env_id']}_{task_data['mode']}_{self.cur_decision_step}.txt")
            self.initial_collect(traj_file_path, task_data)
            
            if self.cur_decision_step > self.max_decision_step:
                terminate_info = {'terminate': 'max_step', 'decision_step': self.cur_decision_step}
                self.write_text_traj(traj_file_path, f'Max decision steps reached.\n')
                return terminate_info
            
            skill_set = self.env.get_skill_set()

            try:
                prompt_obs = self.make_observation(observation, working_memory.scene_graph, previous_step)
                print(prompt_obs)
                self.llm_agent.reset(target_info, self.history)
                self.write_text_traj(traj_file_path, prompt_obs)
                self.llm_agent.add_text_traj(prompt_obs)
                next_step_info = self.llm_agent.decision_making(skill_set)
                next_step_class, next_step, reasoning = next_step_info['next_step_class'], next_step_info['next_step'], next_step_info['reasoning']
                self.write_text_traj(traj_file_path, f'Think: {reasoning}\n')
                self.write_text_traj(traj_file_path, f'{next_step_class}: {next_step}\n')
            except Exception as e:
                terminate_info = {'terminate': 'plan_next_step_error', 'decision_step': self.cur_decision_step}
                self.write_text_traj(traj_file_path, f"Plan Next Step Error: {e}\n")
                return terminate_info
            
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
                    working_memory.update_nx_graph(sim_graph, next_step.split('go to ')[-1])    
                else:
                    working_memory.update_nx_graph(sim_graph)
                if obs['success']:
                    observation = obs['obs_text']
                    previous_step = (next_step, obs['success'])
                else:
                    observation = obs['feedback']
                    previous_step = (next_step, obs['success'])
                self.cur_decision_step += 2
                print(observation)
            print(f'{self.cur_decision_step}: {next_step_class}: {next_step}')

    def make_target_info(self):
        raise NotImplementedError("make_target_info() is not implemented.")
    
    def get_result(self, task_d):
        raise NotImplementedError()
    
    def make_observation(self, observation, scene_graph, previous_step):
        if previous_step is not None:
            self.history.append(previous_step)
            if len(self.history) > 50:
                self.history.pop(0)
        obs = observation + ' Furthermore, you have found the following rooms and objects in the house so far (areas in parentheses have not been explored yet):\n'
        room_object = self.make_list_room_object(scene_graph)
        obs += room_object + '\n'
        if self.history:
            formatted_history = ', '.join([
                f"{step} (success)" if success else f"{step} (fail)" 
                for step, success in self.history
            ])
            obs += f"Your {len(self.history)} previous actions were: {formatted_history}\n"
        else:
            obs += "You have not taken any actions yet.\n"
        obs += f"What is the best next action to complete the task as efficiently as possible?\n"
        return obs
    
    def make_list_room_object(self, scene_graph):
        room_to_objects = defaultdict(list)
        room_node_map = dict()  # name_id_nl → room_node
        # (1) room 노드 수집
        room_nodes = [n for n, data in scene_graph.nodes(data=True) if data.get('node_type') == 'room']
        # (2) 각 room 노드에 대해 들어오는 INSIDE edge 탐색
        for room_node in room_nodes:
            room_data = scene_graph.nodes[room_node]
            room_name_raw = room_data.get("name_id_nl", room_node)
            room_node_map[room_name_raw] = room_node
            for source_node, _, edge_data in scene_graph.in_edges(room_node, data=True):
                if edge_data.get("relation") != "INSIDE":
                    continue
                source_data = scene_graph.nodes[source_node]
                obj_name_raw = source_data.get("name_id_nl", source_node)
                if 'user' in obj_name_raw[0].lower():
                    continue
                room_to_objects[room_name_raw].append(obj_name_raw)
        # (3) 포맷팅
        formatted_lines = []
        for room_key, obj_list in sorted(room_to_objects.items()):
            room_node = room_node_map.get(room_key, None)
            visited = scene_graph.nodes[room_node].get("visited", False) if room_node else False
            # room name format
            if isinstance(room_key, tuple) and len(room_key) == 2:
                name, id_ = room_key
                room_str = f"({name} {id_})" if not visited else f"{name} {id_}"
            else:
                room_str = str(room_key)
            # object format
            obj_strs = []
            # for obj_key in obj_list:
            for obj_key in sorted(obj_list, key=lambda x: str(x).lower()):
                if isinstance(obj_key, tuple) and len(obj_key) == 2:
                    name, id_ = obj_key
                    obj_str = None
                    for node_id, node_data in scene_graph.nodes(data=True):
                        if node_data.get("name_id_nl") == obj_key:
                            node_type = node_data.get("node_type", "")
                            visited = node_data.get("visited", False)
                            # 무조건 괄호 없이 출력 if node_type == object
                            if node_type == "object":
                                obj_str = f"{name} {id_}"
                            else:
                                obj_str = f"({name} {id_})" if not visited else f"{name} {id_}"
                            break
                    if obj_str is None:
                        obj_str = f"{name} {id_}"
                else:
                    obj_str = str(obj_key)
                obj_strs.append(obj_str)
            formatted_line = f"- {room_str}: [{', '.join(obj_strs)}]"
            formatted_lines.append(formatted_line)
        return "\n".join(formatted_lines)


