import os, math, re
import json
import sys
import copy
import numpy as np
from scipy import spatial
from ai2thor.controller import Controller
import core.utils.procthor_utils as procthor_utils
from shapely.geometry import Point, Polygon


from core.env.base_env import BaseEnv


class ProcThorEnv(BaseEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.env_id = None
        self.mode = None
        self.task_goal = None
        self.init_state = None
        self.agent_id = None
        
        self.envs = procthor_utils.load_dataset()["test"]
        self.controller = Controller(scene=self.envs[0])
        self.agent_height = 0.9
        self.CAMERA_HEIGHT_OFFSET = 0.75
        self.on_recep = ['armchair', 'bed', 'chair', 'coffeemachine', 'coffeetable', 'countertop', 'desk', 'diningtable', 'dresser', 'handtowelholder', 'laundryhamper', 'ottoman', 'pan', 'plate', 'shelf', 'sidetable', 'sofa', 'stoveburner', 'toiletpaperhanger', 'towelholder', 'tvstand', 'stool', 'toilet']
        self.in_recep = ['bathtub', 'bathtubbasin', 'box', 'cabinet', 'cup', 'drawer', 'fridge', 'garbagecan', 'microwave', 'mug', 'pot', 'safe', 'sink', 'sinkbasin', 'toaster', 'washingmachine', 'bowl']
        
        # object name dicts
        with open(cfg.procthor.obj_dict_sim2nl, 'r') as f:
            self.obj_dict_sim2nl = json.load(f)
        with open(cfg.procthor.obj_dict_nl2sim, 'r') as f:
            self.obj_dict_nl2sim = json.load(f)
            
        self.filtered_sim_objs = cfg.procthor.filtered_sim_objs
        self.sim_receps = cfg.procthor.sim_receps
        self.sim_receps_sg = cfg.procthor.sim_receps_for_sg
        print("[PROCTHOR] ProcThorEnv initialized.")

    def reset(self, task_d):
        # Task 초기화
        self.env_id = task_d["env_id"]
        self.mode = task_d["mode"]
        self.task_goal = task_d["task_goal"]
        self.init_state = task_d.get("init_state", [])
        self.reset_states()
        self.cur_recep_info = (None, None)
        self.sliced = []
                
        # Scene 초기화
        house = self.envs[self.env_id]
        self.controller.reset(scene=house)
        
        if self.init_state:
            self.set_init_state(self.mode, self.init_state)
        
        self.init_event = self.controller.step(action="Pass")  
        self.last_event = self.controller.step(action="Pass")  
        nodes = self.last_event.metadata.get("objects", [])
        self.nodes = procthor_utils.filter_ignored_nodes(nodes, self.sliced)
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        
        # room 정보 추출 및 polygon 저장
        self.room_info = {}  # room_id -> {type, polcygon (shapely.Polygon)}
        if "rooms" in house:
            for room in house["rooms"]:
                room_id = room["id"]
                room_type = room.get("roomType", "Unknown").lower()
                floor_poly = Polygon([(p["x"], p["z"]) for p in room.get("floorPolygon", [])])
                self.room_info[room_id] = {
                    "roomType": room_type,
                    "polygon": floor_poly
                }
        
        # name-id 변환
        self.name_id_dict_sim2nl, self.name_id_dict_nl2sim = procthor_utils.make_name_id_dict(self.nodes, self.obj_dict_sim2nl, self.room_info)

        # self.vis_log = [{'action': 'init', 'images': self.get_visual_obs()}]
        self.working_memory = {}
        
        obs_text = self.get_text_obs()
        obs_vis = self.get_visual_obs()
        results = {
            "obs_text": obs_text,
            "obs_vis": obs_vis
        }

        # self.vis_log = [{'action': 'Init', 'images': self.get_visual_obs()[0], 'observation': obs_text}]
        self.vis_log = [{'action': f"User Instruction: {task_d['instruction'][0]}", 'images': self.get_visual_obs()[0], 'observation': obs_text}]
        return results
    
    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()
        self.filled_coffee_objects = set()
    
    def set_init_state(self, mode, object_ids):
        action = "ToggleObjectOn"
        
        for obj in object_ids:
            event = self.controller.step(
                action=action,
                objectId=obj,
                forceAction=True
            )
            success = event.metadata.get("lastActionSuccess", False)   
            if not success:
                print("Error: cannot turn on objects in set_init_state")
    
    def get_reachable_positions(self):
        free_positions = self.controller.step(dict(action="GetReachablePositions")).metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']] for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        return free_positions, kd_tree
    
    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]
    
    def step(self, nl_action):
        sim_skill_info = procthor_utils.decompose_nl_skill(nl_action, self.name_id_dict_nl2sim)
        sim_act, sim_obj_info = sim_skill_info['sim_act'], sim_skill_info['sim_obj_info']
        if sim_act is None and sim_obj_info is None:
            results = {'success': False,
                   'feedback': "Invalid action.",
                   'obs_text': None,
                   'obs_vis': self.get_visual_obs(),
                   'reason_type': 'precondition',
                   }
            return results
        elif sim_act is None:
            results = {'success': False,
                   'feedback': f"You have never seen {sim_obj_info[0]} {sim_obj_info[1]} before.",
                   'obs_text': None,
                   'obs_vis': self.get_visual_obs(),
                   'reason_type': 'precondition',
                   }
            return results
        if sim_act == 'TeleportFull':
            success, feedback = self.nav_obj(sim_act, sim_obj_info)
            if success and (sim_obj_info[0] in self.sim_receps):
                self.cur_recep_info = sim_obj_info
        elif sim_act == 'PickupObject':
            success, feedback = self.pick(sim_act, sim_obj_info)
        elif sim_act == 'PutObject':
            success, feedback = self.put(sim_act, sim_obj_info)
        elif sim_act == 'OpenObject':
            success, feedback = self.open(sim_act, sim_obj_info)
            if success and (sim_obj_info[0] in self.sim_receps):
                self.cur_recep_info = sim_obj_info
        elif sim_act == 'CloseObject':
            success, feedback = self.close(sim_act, sim_obj_info)
            if success and (sim_obj_info[0] in self.sim_receps):
                self.cur_recep_info = sim_obj_info
        elif sim_act == 'ToggleObjectOn':
            success, feedback = self.toggleon(sim_act, sim_obj_info)
        elif sim_act == 'ToggleObjectOff':
            success, feedback = self.toggleoff(sim_act, sim_obj_info)
        elif sim_act == 'SliceObject':
            success, feedback = self.sliceobj(sim_act, sim_obj_info)
        elif sim_act == 'DropHandObject':
            success, feedback = self.drop(sim_act)
            
        if success:
            obs_text = self.get_text_obs(nl_action)
        else:
            obs_text = None

        obs_vis = self.get_visual_obs()
        
        if not success and "controller" in feedback:
            reason_type = 'controller'
            vis_log_text = f'{nl_action} (execution fail)'
        elif not success:
            reason_type = 'precondition'
            vis_log_text = f'{nl_action} (fail)'
        else:
            reason_type = None
            vis_log_text = nl_action
            
        self.vis_log.append({'action': vis_log_text, 'images': obs_vis[0], 'observation': obs_text})
        
        results = {'success': success,
                   'feedback': feedback,
                   'obs_text': obs_text,
                   'obs_vis': obs_vis,
                   'reason_type': reason_type,
                   }
        
        return results
            
    def drop(self, action):
        success = False
        feedback = ''
        self.last_event = self.controller.step(dict(
            action=action,
            forceAction=True
        ))
        if not self.last_event.metadata['lastActionSuccess']:
            if len(self.last_event.metadata['inventoryObjects']) == 0:
                feedback = f'Drop action failed: Robot is not holding any object.'
            else:
                feedback = f"(controller) Drop action failed: {self.last_event.metadata['errorMessage']}"
        else:
            success = True
        return success, feedback
            
    def sliceobj(self, action, obj_info):
        obj_data, obj_id = obj_info
        success = False
        feedback = ''
        if obj_id is None:
            feedback = f'Slice action failed: Cannot find {obj_data.lower()} to slice.'
            return success, feedback
        nl_obj_info = self.name_id_dict_sim2nl[obj_info]
        inventory = self.last_event.metadata.get("inventoryObjects", [])
        if not any(obj["objectType"] in {"Knife", "ButterKnife"} for obj in inventory):
            feedback = "Slice action failed: You're not holding anything that can be used to slice."
            return success, feedback
        
        self.last_event = self.controller.step(dict(
            action=action,
            objectId=obj_id,
        ))
        if not self.last_event.metadata['lastActionSuccess']:
            feedback = f"(controller) Slice action failed: {self.last_event.metadata['errorMessage']}"
            if "Target object not found within the specified visibility" in feedback:
                feedback = f"Slice action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) is not within the visible range."
            return success, feedback
        
        success = True
            
        sliced_obj = None
        for obj in self.last_event.metadata['objects']:
            if obj.get('objectType', '').lower() == obj_data + 'sliced' and obj['objectId'].startswith(obj_id + "|"):
                sliced_obj = obj
                break
            
        if sliced_obj:
            sliced_id = sliced_obj['objectId']
            sliced_type = sliced_obj['objectType'].lower()
            nl_name = self.obj_dict_sim2nl[sliced_type]
            sim_key = (sliced_type, sliced_id)
            existing_indices = [
                nl_key[1]
                for nl_key in self.name_id_dict_nl2sim.keys()
                if nl_key[0] == nl_name
            ]
            next_index = max(existing_indices, default=0) + 1
            
            nl_key = (nl_name, 1)
            self.name_id_dict_sim2nl[sim_key] = nl_key
            self.name_id_dict_nl2sim[nl_key] = sim_key
            self.sliced.append(obj_id)
            if obj_id in self.heated_objects:
                self.heated_objects.add(sliced_id)
            if obj_id in self.cooled_objects:
                self.cooled_objects.add(sliced_id)
        return success, feedback
            
    def toggleoff(self, action, obj_info):
        obj_data, obj_id = obj_info
        success = False
        feedback = ''
        if obj_id is None:
            feedback = f'Turn off action failed: Cannot find {obj_data.lower()} to turn off.'
        else:
            nl_obj_info = self.name_id_dict_sim2nl[obj_info]
            self.last_event = self.controller.step(dict(
                action=action,
                objectId=obj_id,
            ))
            if not self.last_event.metadata['lastActionSuccess']:
                feedback = f"(controller) Turn off action failed: {self.last_event.metadata['errorMessage']}"
                if "Target object not found within the specified visibility" in feedback:
                    feedback = f"Turn off action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) is not within the visible range."
            else:
                success = True
        return success, feedback
            
    def toggleon(self, action, obj_info):
        obj_data, obj_id = obj_info
        success = False
        feedback = ''
        if obj_id is None:
            feedback = f'Turn on action failed: Cannot find {obj_data.lower()} to turn on.'
            return success, feedback
        self.last_event = self.controller.step(dict(
            action=action,
            objectId=obj_id,
        ))
        nl_obj_info = self.name_id_dict_sim2nl[obj_info]
        if not self.last_event.metadata['lastActionSuccess']:
            err_msg = self.last_event.metadata['errorMessage']
            if "Target object not found within the specified visibility" in err_msg:
                feedback = f"Turn on action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) is not within the visible range."
            else:
                feedback = f"(controller) Turn on action failed: {err_msg}"
            return success, feedback
        # Success case
        success = True
        objects = self.last_event.metadata.get("objects", [])
        # Microwave 내부 오브젝트를 heated_objects에 추가
        if "Microwave" in obj_id:
            microwave_obj = next((obj for obj in objects if obj.get("objectId") == obj_id), None)
            if microwave_obj:
                heated_ids = microwave_obj.get("receptacleObjectIds", [])
                self.heated_objects.update(heated_ids)
        if "Toaster" in obj_id:
            toaster_obj = next((obj for obj in objects if obj.get("objectId") == obj_id), None)
            if toaster_obj:
                heated_ids = toaster_obj.get("receptacleObjectIds", [])
                self.heated_objects.update(heated_ids)
        # Faucet 사용 시 가까운 SinkBasin 안의 오브젝트를 cleaned_objects에 추가
        elif "Faucet" in obj_id:
            faucet_obj = next((obj for obj in objects if obj.get("objectId") == obj_id), None)
            if faucet_obj:
                fx, fz = faucet_obj["position"]["x"], faucet_obj["position"]["z"]
                closest = min(
                    (obj for obj in objects if obj.get("objectType") == "SinkBasin"),
                    key=lambda o: ((fx - o["position"]["x"]) ** 2 + (fz - o["position"]["z"]) ** 2),
                    default=None
                )
                if closest:
                    self.cleaned_objects.update(closest.get("receptacleObjectIds", []))
        elif "CoffeeMachine" in obj_id:
            coffeemachine_obj = next((obj for obj in objects if obj.get("objectId") == obj_id), None)
            if coffeemachine_obj:
                coffee_ids = coffeemachine_obj.get("receptacleObjectIds", [])
                self.filled_coffee_objects.update(coffee_ids)
        return success, feedback
            
    def close(self, action, obj_info):
        obj_data, obj_id = obj_info
        success = False
        feedback = ''
        if obj_id is None:
            feedback = f'Close action failed: Cannot find {obj_data.lower()} to close.'
        else:
            nl_obj_info = self.name_id_dict_sim2nl[obj_info]
            self.last_event = self.controller.step(dict(
                action=action,
                objectId=obj_id,
            ))
            if not self.last_event.metadata['lastActionSuccess']:
                feedback = f"(controller) Close action failed: {self.last_event.metadata['errorMessage']}"
                if "Target object not found within the specified visibility" in feedback:
                    feedback = f"Close action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) is not within the visible range."
            else:
                success = True
                if "Fridge" in obj_id:
                    objects = self.last_event.metadata.get("objects", [])
                    fridge_obj = next((obj for obj in objects if obj.get("objectId") == obj_id), None)
                    if fridge_obj:
                        cooled_ids = fridge_obj.get("receptacleObjectIds", [])
                        self.cooled_objects.update(cooled_ids)
        return success, feedback
            
    def open(self, action, obj_info):
        obj_data, obj_id = obj_info
        success = False
        feedback = ''

        if obj_id is None:
            feedback = f'Open action failed: Cannot find {obj_data.lower()} to open.'
        else:
            nl_obj_info = self.name_id_dict_sim2nl[obj_info]
            for i in range(4):
                self.last_event = self.controller.step(dict(
                    action=action,
                    objectId=obj_id,
                    forceAction=True
                ))
                if self.last_event.metadata['lastActionSuccess']:
                    success = True
                    return success, feedback

                else:
                    feedback = f"(controller) Open action failed: {self.last_event.metadata['errorMessage']}"
                    if "Target object not found within the specified visibility" in feedback:
                        feedback = f"Open action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) is not within the visible range."
                    if "Target must be OFF to open!" in feedback:
                        feedback = f"Open action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) must be OFF to open."

                    # move around to avoid self-collision
                    if i == 0:
                        self.controller.step(dict(action="MoveBack"))
                    elif i == 1:
                        self.controller.step(dict(action="MoveBack"))
                        self.controller.step(dict(action="MoveRight"))
                    elif i == 2:
                        self.controller.step(dict(action="MoveLeft"))
                        self.controller.step(dict(action="MoveLeft"))

        return success, feedback
            
    def put(self, action, obj_info):
        if self.cur_recep_info[0] is None:
            success, feedback = self.drop('DropHandObject')
            return success, feedback
        obj_data, obj_id = obj_info
        obj_data_nl, obj_id_nl = self.name_id_dict_sim2nl[obj_info]
        recep_data, recep_id = self.cur_recep_info
        recep_data_nl, recep_id_nl = self.name_id_dict_sim2nl[self.cur_recep_info]
        success = False
        feedback = ''

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            feedback = f'Put action failed: Robot is not holding any object.'
            return success, feedback
        
        holding_data = self.last_event.metadata['inventoryObjects'][0]['objectId']
        
        if holding_data != obj_id:
            feedback = f'Put action failed: Robot is not holding {obj_data_nl} ({obj_id_nl}).'
            return success, feedback
        else:
            holding_obj_id = holding_data
            
        for j in range(7):
            if j == 1:
                self.controller.step(dict(action="LookUp"))
                self.controller.step(dict(action="LookUp"))
            elif j == 2:
                for _ in range(4):
                    self.controller.step(dict(action="LookDown"))
            elif j == 3:
                self.controller.step(dict(action="LookUp"))
                self.controller.step(dict(action="LookUp"))
                self.controller.step(dict(action="MoveBack"))
            elif j == 4:
                self.controller.step(dict(action="MoveAhead"))
                for _ in range(4):
                    self.controller.step(dict(action="MoveRight"))
            elif j == 5:
                for _ in range(8):
                    self.controller.step(dict(action="MoveLeft"))
            elif j == 6:
                for _ in range(4):
                    self.controller.step(dict(action="MoveRight"))
                self.controller.step(dict(action="RotateHand", x=40, y=0, z=0))

            self.last_event = self.controller.step(dict(
                action=action,
                objectId=recep_id,
                forceAction=True
            ))

            if self.last_event.metadata['lastActionSuccess']:
                success = True
                feedback = ''
                break
            else:
                feedback = f"(controller) Put action failed: {self.last_event.metadata['errorMessage']}"
                if "No valid positions to place object found" in feedback:
                    feedback = f"Put action failed: There is no valid space to place {obj_data_nl} ({obj_id_nl}) in {recep_data_nl} ({recep_id_nl})."
                if "Target openable Receptacle is CLOSED" in feedback:
                    feedback = f"Put action failed: {recep_data_nl} ({recep_id_nl}) is closed."
                if "not a valid Object Type to be placed" in feedback:
                    feedback = f"Put action failed: {obj_data_nl} ({obj_id_nl}) is not a valid Object Type to be placed in {recep_data_nl} ({recep_id_nl})."
        return success, feedback
            
    def pick(self, action, obj_info):
        obj_data, obj_id = obj_info
        success = False
        feedback = ''
        if obj_id is None:
            return False, f'Pick up action failed: Cannot find {obj_data.lower()} to pick up'
        nl_obj_info = self.name_id_dict_sim2nl[obj_info]
        objects = self.last_event.metadata.get("objects", [])
        obj = next((o for o in objects if o['objectId'] == obj_id), None)
        if obj is None:
            return False, f'Pick up action failed: Object {obj_data.lower()} not found in metadata'
        parent_receptacles = obj.get("parentReceptacles") or []
        if parent_receptacles:
            for recep_id in parent_receptacles:
                if "toilet" not in recep_id.lower():
                    recep_obj = next((o for o in objects if o['objectId'] == recep_id), None)
                    if recep_obj and recep_obj.get("openable", False) and not recep_obj.get("isOpen", True):
                        recep_type = recep_obj["objectType"].lower()
                        recep_data_nl, recep_id_nl = self.name_id_dict_sim2nl[(recep_obj["objectType"].lower(), recep_obj["objectId"])]
                        return False, f"Pick up action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) is inside a closed {recep_data_nl} ({recep_id_nl})."
        for j in range(7):
            if j == 1:
                self.controller.step(dict(action="LookUp"))
                self.controller.step(dict(action="LookUp"))
            elif j == 2:
                for _ in range(5):
                    self.controller.step(dict(action="LookDown"))
            elif j == 3:
                self.controller.step(dict(action="LookUp"))
                self.controller.step(dict(action="LookUp"))
                self.controller.step(dict(action="LookUp"))
                self.controller.step(dict(action="MoveBack"))
                self.controller.step(dict(action="MoveBack"))
                self.controller.step(dict(action="MoveBack"))
            elif j == 4:
                self.controller.step(dict(action="MoveAhead"))
                self.controller.step(dict(action="MoveAhead"))
                self.controller.step(dict(action="MoveAhead"))
                for _ in range(4):
                    self.controller.step(dict(action="MoveRight"))
            elif j == 5:
                for _ in range(8):
                    self.controller.step(dict(action="MoveLeft"))
            elif j == 6:
                for _ in range(4):
                    self.controller.step(dict(action="MoveRight"))
                self.controller.step(dict(action="RotateHand", x=40, y=0, z=0))

            # pick 시도
            self.last_event = self.controller.step(dict(
                action=action,
                objectId=obj_id,
                forceAction=False
            ))

            if self.last_event.metadata['lastActionSuccess']:
                success = True
                feedback = ''
                break
            else:
                inventory = self.last_event.metadata.get('inventoryObjects', [])
                if not inventory:
                    feedback = f'(controller) Pick up action failed: {self.last_event.metadata["errorMessage"]}'
                    if "Target object not found within the specified visibility" in feedback:
                        feedback = f"Pick up action failed: {nl_obj_info[0]} ({nl_obj_info[1]}) is not within the visible range."
                else:
                    held = inventory[0]
                    feedback = f'Pick up action failed: Robot is currently holding {held["objectType"].lower()}.'
                    break  # 이미 다른 물건을 들고 있으므로 pick 반복 중단

        return success, feedback

        
    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))
    def nav_obj(self, action, obj_info):
        objects = self.last_event.metadata.get("objects", [])
        obj_data, obj_id = obj_info
        teleport_success = False
        feedback = ''
        
        # 방 이름이면 room teleport 로직 실행
        room_types = {'kitchen', 'livingroom', 'bedroom', 'bathroom'}
        if obj_data in room_types:
            room_id = obj_id  # 여기선 obj_id가 실제로 room_id임
            room_polygon = self.room_info[room_id]['polygon']
        
            # Get reachable positions
            positions = self.controller.step(dict(action="GetReachablePositions")).metadata["actionReturn"]
            if not positions:
                feedback = f"GetReachablePositions failed or returned no data in room {obj_data.lower()}"
                return teleport_success, feedback
                
            inside_points = [
                p for p in positions if room_polygon.contains(Point(p['x'], p['z']))
            ]

            if not inside_points:
                feedback = f"No reachable points found inside room {obj_data.lower()}"
                return teleport_success, feedback

            # 가장 중앙에 가까운 위치 선택
            center = room_polygon.centroid
            inside_points.sort(key=lambda p: Point(p['x'], p['z']).distance(center))
            target = inside_points[0]

            self.last_event = self.controller.step(dict(
                action=action,
                x=target['x'],
                y=self.agent_height,
                z=target['z'],
                rotation=0,
                horizon=0,
                standing=True
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                feedback = f"(controller) Teleport to room {obj_data} failed: {self.last_event.metadata['errorMessage']}"
            else:
                teleport_success = True
            return teleport_success, feedback
     
        # find object index from id
        obj_idx = -1
        for i, o in enumerate(objects):
            if o['objectId'] == obj_id:
                obj_idx = i
                break
        
        if obj_idx == -1:
            feedback = f'Cannot find {obj_data.lower()}'
            
        else:
            # teleport sometimes fails even with reachable positions. if fails, repeat with the next closest reachable positions.
            max_attempts = 20

            # get obj location
            loc = objects[obj_idx]['position']
            obj_rot = objects[obj_idx]['rotation']['y']

            # do not move if the object is already visible and close
            if objects[obj_idx]['visible'] and objects[obj_idx]['distance'] < 1.0:
                max_attempts = 0
                teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (obj_data == 'fridge' or obj_data == 'microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position([loc['x'], loc['y'], loc['z']], reachable_pos_idx)

                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(loc['x'] - closest_loc[0]), loc['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (obj_data == 'fridge' or obj_data == 'microwave'):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if obj_data == 'fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if obj_data == 'microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + self.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(loc['x'] - closest_loc[0], loc['z'] - closest_loc[2])
                hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport
                self.last_event = self.controller.step(dict(action=action,
                                  x=closest_loc[0], y=self.agent_height, z=closest_loc[2],
                                  rotation=rot_angle, horizon=-hor_angle, standing=True))

                if not self.last_event.metadata['lastActionSuccess']:
                    feedback = (
                        f"(controller) TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                else:
                    teleport_success = True
                    break

            if not teleport_success:
                feedback = f'(controller) TeleportFull action failed: {self.last_event.metadata["errorMessage"]}'

        return teleport_success, feedback
    
    def get_visual_obs(self):
        frame = self.last_event.frame
        return [frame]

    def get_text_obs(self, nl_action=None):
        nodes = self.last_event.metadata.get("objects", [])
        self.nodes = procthor_utils.filter_ignored_nodes(nodes, self.sliced)
        partial_graph, obs_agent_room, obs_all_rooms = self.get_partial_nodes()
        if partial_graph == None:
            return "You can't see anything."
        
        if nl_action == None:
            obs_room_items = procthor_utils.obs_room_items(partial_graph, self.last_event)
            obs_agent_room_nl = self.name_id_dict_sim2nl[obs_agent_room]
            obs_text = f'You are in the house, and there are {len(obs_all_rooms)} rooms: {procthor_utils.merge_obs_list(obs_all_rooms, self.name_id_dict_sim2nl)}. '
            obs_text += f'You are in the middle of a {obs_agent_room_nl[0]} ({obs_agent_room_nl[1]}). '
            obs_text += f'Looking quickly around the room, you see {procthor_utils.merge_obs_list(obs_room_items, self.name_id_dict_sim2nl)}.'
        else:
            sim_skill_info = procthor_utils.decompose_nl_skill(nl_action, self.name_id_dict_nl2sim)
            sim_act, sim_obj_info = sim_skill_info['sim_act'], sim_skill_info['sim_obj_info']
            nl_obj_info = self.name_id_dict_sim2nl[sim_obj_info]
            
            if sim_act == 'TeleportFull':
                if sim_obj_info[0] in ['kitchen', 'livingroom', 'bedroom', 'bathroom']:
                    obs_room_items = procthor_utils.obs_room_items(partial_graph, self.last_event)
                    obs_text = f'You move to the {nl_obj_info[0]} ({nl_obj_info[1]}). '
                    obs_text += f'Looking quickly around the room, you see {procthor_utils.merge_obs_list(obs_room_items, self.name_id_dict_sim2nl)}.'
                else:
                    obs_text = f'You arrive at the {nl_obj_info[0]} ({nl_obj_info[1]}). '
                    if procthor_utils.check_attributes(self.nodes, sim_obj_info[1], 'openable'):
                        if not procthor_utils.check_attributes(self.nodes, sim_obj_info[1], 'isOpen'):
                            obs_text += f'The {nl_obj_info[0]} ({nl_obj_info[1]}) is closed. '
                        elif procthor_utils.check_attributes(self.nodes, sim_obj_info[1], 'isOpen'):
                            obs_text += f'The {nl_obj_info[0]} ({nl_obj_info[1]}) is open. '
                    if sim_obj_info[0] in self.sim_receps:
                        obs_close_objs = procthor_utils.obs_close_objs_recep(partial_graph, self.last_event, self.cur_recep_info)
                        if obs_close_objs:
                            obs_text += f'You see {procthor_utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}.'
                    else:
                        obs_close_objs = procthor_utils.obs_close_objs(partial_graph, self.last_event, self.cur_recep_info)
                        if obs_close_objs:
                            obs_text += f'You see {procthor_utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}.'
            elif sim_act == 'PickupObject':
                obs_text = f'You pick up {nl_obj_info[0]} ({nl_obj_info[1]}).'
            elif sim_act == 'PutObject':
                sim_recep_info = self.cur_recep_info
                if sim_recep_info[0] is None:
                    obs_text = f"You put {nl_obj_info[0]} ({nl_obj_info[1]})."
                else:
                    nl_recep_info = self.name_id_dict_sim2nl[sim_recep_info]
                    if sim_recep_info[0] in self.on_recep:
                        obs_text = f"You put {nl_obj_info[0]} ({nl_obj_info[1]}) on {nl_recep_info[0]} ({nl_recep_info[1]})."
                    else:
                        obs_text = f"You put {nl_obj_info[0]} ({nl_obj_info[1]}) in {nl_recep_info[0]} ({nl_recep_info[1]})."
            elif sim_act == 'OpenObject':
                obs_text = f'You open {nl_obj_info[0]} ({nl_obj_info[1]}). '
                obs_close_objs = procthor_utils.obs_close_objs_recep(partial_graph, self.last_event, self.cur_recep_info)
                if obs_close_objs:
                    obs_text += f'You see {procthor_utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}.'
            elif sim_act == 'CloseObject':
                obs_text = f'You close {nl_obj_info[0]} ({nl_obj_info[1]}).'
            elif sim_act == 'ToggleObjectOn':
                obs_text = f'You turn on {nl_obj_info[0]} ({nl_obj_info[1]}).'
            elif sim_act == 'ToggleObjectOff':
                obs_text = f'You turn off {nl_obj_info[0]} ({nl_obj_info[1]}).'
            elif sim_act == 'SliceObject':
                obs_text = f'You slice {nl_obj_info[0]} ({nl_obj_info[1]}). '
                obs_close_objs = procthor_utils.obs_close_objs(partial_graph, self.last_event, self.cur_recep_info)
                if obs_close_objs:
                    obs_text += f'You see {procthor_utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}.'
            elif sim_act == 'DropHandObject':
                obs_text = f'You drop {nl_obj_info[0]} ({nl_obj_info[1]}).'
        obs_grab_objs = procthor_utils.obs_agent_grab(self.last_event)
        if not obs_grab_objs == None:
            obs_text += f' You hold {procthor_utils.merge_obs_list(obs_grab_objs, self.name_id_dict_sim2nl)}.'
        return obs_text

    def get_partial_nodes(self):        
        agent_position = self.last_event.metadata["agent"]["position"]
        agent_point = Point(agent_position["x"], agent_position["z"])
        
        # Step 1: room nodes + polygon 찾기
        room_id = None
        room_name = None
        all_rooms = []
        
        for rid, info in self.room_info.items():
            all_rooms.append((info["roomType"], rid))
            if info["polygon"].contains(agent_point):
                room_id = rid
                room_name = info["roomType"]

        if room_id is None:
            print("[WARNING] Agent is not inside any known room polygon.")
            return None, None, None
        
        # Step 2: 해당 방 안에 포함된 object 탐색
        room_objects = []
        all_room_objs = [] 
                        
        closed_receptacle_ids = {
            obj["objectId"] for obj in self.nodes
            if obj["openable"] and (not obj["isOpen"]) and obj["receptacle"]
        }

        for obj in self.nodes:
            obj_id = obj["objectId"]
            obj_type = obj["objectType"]
            obj_pos = obj["position"]
            obj_point = Point(obj_pos["x"], obj_pos["z"])

            # 이 object가 현재 room polygon 내부에 있어야 함
            if not self.room_info[room_id]["polygon"].contains(obj_point):
                continue

            # parentReceptacles에 closed된 receptacle이 있으면 제외
            parent_receptacles = obj.get("parentReceptacles") or []
            if any(pid in closed_receptacle_ids for pid in parent_receptacles if "toilet" not in pid.lower()):
                continue

            room_objects.append(obj)

        # 현재 방의 room node도 obs_agent_room으로 리턴
        room_node = (room_name, room_id)

        return room_objects, room_node, all_rooms
    
    def get_graph_obs(self, visibility='full'):
        if visibility == 'full':
            return self.get_graph()
        elif visibility == 'partial':
            return self.get_partial_graph()
        elif visibility == 'initial':
            return self.get_init_graph()
        else:
            raise NotImplementedError(f"Visibility type '{visibility}' is not implemented.")
    
    def get_graph(self):
        nodes = self.last_event.metadata.get("objects", [])
        self.nodes = procthor_utils.filter_ignored_nodes(nodes, self.sliced)
        return self.nodes
    
    def get_partial_graph(self):
        full_graph = self.get_graph()
        nodes = self.last_event.metadata.get("objects", [])
        self.nodes = procthor_utils.filter_ignored_nodes(nodes, self.sliced)
        objects, room_node, all_rooms = self.get_partial_nodes()
        if objects == None: 
            return {"nodes": [], "edges": []}
            
        nodes = []
        edges = []
        excluded_pairs = set()
        # Step 1: Add all room nodes
        node_id_counter = 0
        room_id_to_node_id = {}
        for room_type, room_id in all_rooms:
            node = {
                "id": node_id_counter,
                "objectType": room_type.lower(),
                "objectId": room_id,
                "category": "Rooms"
            }
            room_id_to_node_id[room_id] = node_id_counter
            nodes.append(node)
            node_id_counter += 1
        # Step 2: Add all object nodes
        id_map = {}
        for idx, obj in enumerate(objects):
            node_id = idx + node_id_counter
            obj_type = obj["objectType"]
            id_map[obj["objectId"]] = node_id
            node = {
                "id": node_id,
                "category": "asset" if obj_type in self.sim_receps_sg else "object"
            }
            node.update(obj)
            nodes.append(node)
        # User node
        user_node_id = len(nodes)
        user_pos = self.last_event.metadata["agent"]["position"]
        user_node = {
            "id": user_node_id,
            "objectType": "user",
            "objectId": "agent",
            "category": "user",
            "position": user_pos
        }
        nodes.append(user_node)
        self.agent_id = user_node_id
        # Step 3: Add on/in edges
        object_lookup = {obj["objectId"]: obj for obj in objects}
        for obj in objects:
            from_id = id_map[obj["objectId"]]
            for re_id in obj.get("parentReceptacles") or []:
                to_id = id_map.get(re_id)
                if to_id is None:
                    continue
                to_obj = object_lookup.get(re_id)
                if to_obj is None:
                    continue
                receptacle_type = to_obj.get("objectType", "")
                relation = "INSIDE" if receptacle_type in self.in_recep else "ON"
                edges.append({
                    "from_id": from_id,
                    "to_id": to_id,
                    "relation_type": relation
                })
                excluded_pairs.add((from_id, to_id))
                excluded_pairs.add((to_id, from_id))
        # Step 4: Add close edges
        all_objs_with_user = objects + [user_node]  
        id_map["agent"] = user_node_id
        for i, obj1 in enumerate(all_objs_with_user):
            id1 = id_map[obj1["objectId"]]
            pos1 = obj1["position"]
            for j, obj2 in enumerate(all_objs_with_user):
                id2 = id_map[obj2["objectId"]]
                if i == j or (id1, id2) in excluded_pairs:
                    continue
                pos2 = obj2["position"]
                if procthor_utils.get_distance(pos1, pos2) < 2.0:
                    edges.append({
                        "from_id": id1,
                        "to_id": id2,
                        "relation_type": "CLOSE"
                    })
        # Step 5: Add room-object edges
        room_type, room_id = room_node
        room_node_id = room_id_to_node_id.get(room_id)
        if room_node_id is not None:
            for obj in all_objs_with_user:
                obj_id = id_map[obj["objectId"]]
                edges.append({
                    "from_id": obj_id,
                    "to_id": room_node_id,
                    "relation_type": "INSIDE"
                })
        # Step 6: Add InFront edge (current receptacle interaction)
        if self.cur_recep_info is not None:
            recep_obj_type, recep_obj_id = self.cur_recep_info
            recep_node_id = id_map.get(recep_obj_id, None)
            if recep_node_id is not None:
                edges.append({
                    "from_id": user_node_id,
                    "to_id": recep_node_id,
                    "relation_type": "InFront"
                })
        return {"nodes": nodes, "edges": edges}
    
    def get_init_graph(self):
        full_graph = self.get_graph()
        nodes = []
        edges = []
        excluded_pairs = set()
        # Step 1: Add all room nodes
        node_id_counter = 0
        room_id_to_node_id = {}
        for rid, info in self.room_info.items():
            room_type = info["roomType"]
            room_id = rid
            node = {
                "id": node_id_counter,
                "objectType": room_type.lower(),
                "objectId": room_id,
                "category": "Rooms"
            }
            room_id_to_node_id[room_id] = node_id_counter
            nodes.append(node)
            node_id_counter += 1
        # Step 2: Add all object nodes
        id_map = {}
        for idx, obj in enumerate(full_graph):
            node_id = idx + node_id_counter
            obj_type = obj["objectType"]
            id_map[obj["objectId"]] = node_id
            node = {
                "id": node_id,
                "category": "asset" if obj_type in self.sim_receps_sg else "object"
            }
            node.update(obj)
            nodes.append(node)
        # User node
        user_node_id = len(nodes)
        user_pos = self.last_event.metadata["agent"]["position"]
        user_node = {
            "id": user_node_id,
            "objectType": "user",
            "objectId": "agent",
            "category": "user",
            "position": user_pos
        }
        nodes.append(user_node)
        self.agent_id = user_node_id
        # Step 3: Add room-object INSIDE edges (polygon 기반)
        obj_room_map = {}
        object_lookup = {obj["objectId"]: obj for obj in full_graph}
        for obj in full_graph:
            obj_id = obj["objectId"]
            obj_pos = obj["position"]
            obj_point = Point(obj_pos["x"], obj_pos["z"])
            obj_node_id = id_map[obj_id]
            matched = False
            for room_id, info in self.room_info.items():
                room_node_id = room_id_to_node_id[room_id]
                polygon = info["polygon"]
                if polygon.contains(obj_point):
                    edges.append({
                        "from_id": obj_node_id,
                        "to_id": room_node_id,
                        "relation_type": "INSIDE"
                    })
                    obj_room_map[obj_id] = room_id
                    matched = True
                    break  # 한 room에만 포함되면 바로 break
            if not matched:
                min_dist = float('inf')
                nearest_room_id = None
                for room_id, info in self.room_info.items():
                    polygon = info["polygon"]
                    dist = polygon.distance(obj_point) 
                    if dist < min_dist:
                        min_dist = dist
                        nearest_room_id = room_id

                if nearest_room_id is not None:
                    room_node_id = room_id_to_node_id[nearest_room_id]
                    edges.append({
                        "from_id": obj_node_id,
                        "to_id": room_node_id,
                        "relation_type": "INSIDE"
                    })
                    obj_room_map[obj_id] = nearest_room_id
                
        # 이제 agent (user node) 도 room과 연결
        agent_point = Point(user_pos["x"], user_pos["z"])
        for room_id, info in self.room_info.items():
            room_node_id = room_id_to_node_id[room_id]
            polygon = info["polygon"]
            if polygon.contains(agent_point):
                edges.append({
                    "from_id": user_node_id,
                    "to_id": room_node_id,
                    "relation_type": "INSIDE"
                })
                obj_room_map["agent"] = room_id
                break
        # Step 4: Add on/in edges using parentReceptacles
        for obj in full_graph:
            from_id = id_map[obj["objectId"]]
            for re_id in obj.get("parentReceptacles") or []:
                to_id = id_map.get(re_id)
                if to_id is None:
                    continue
                to_obj = object_lookup.get(re_id)
                if to_obj is None:
                    continue
                receptacle_type = to_obj.get("objectType", "")
                relation = "INSIDE" if receptacle_type in self.in_recep else "ON"
                edges.append({
                    "from_id": from_id,
                    "to_id": to_id,
                    "relation_type": relation
                })
                excluded_pairs.add((from_id, to_id))
                excluded_pairs.add((to_id, from_id))
        # Step 5: Add CLOSE edges (이제 polygon 없이 room 매핑만 활용)
        id_map["agent"] = user_node_id
        all_objs_with_user = full_graph + [user_node]
        object_lookup["agent"] = user_node
        for i, obj1 in enumerate(all_objs_with_user):
            id1 = id_map[obj1["objectId"]]
            pos1 = obj1["position"]
            room1 = obj_room_map.get(obj1["objectId"], None)
            for j, obj2 in enumerate(all_objs_with_user):
                id2 = id_map[obj2["objectId"]]
                if i == j or (id1, id2) in excluded_pairs:
                    continue
                pos2 = obj2["position"]
                room2 = obj_room_map.get(obj2["objectId"], None)
                if room1 is not None and room1 == room2:
                    if procthor_utils.get_distance(pos1, pos2) < 2.0:
                        edges.append({
                            "from_id": id1,
                            "to_id": id2,
                            "relation_type": "CLOSE"
                        })
        return {"nodes": nodes, "edges": edges}
        
    def get_skill_set(self):
        nodes = self.last_event.metadata.get("objects", [])
        self.nodes = procthor_utils.filter_ignored_nodes(nodes, self.sliced)
        partial_graph, obs_agent_room, obs_all_rooms = self.get_partial_nodes()
        if partial_graph == None:
            return ['done', 'failure']
        obs_partial_objs = procthor_utils.obs_partial_objs(partial_graph) 
        nl_obs_all_rooms_info = [
            self.name_id_dict_sim2nl[obs_room]
            for obs_room in obs_all_rooms
            if obs_room in self.name_id_dict_sim2nl
        ]
        nl_obs_partial_objs_info = [
            self.name_id_dict_sim2nl[partial_obj]
            for partial_obj in obs_partial_objs
            if partial_obj in self.name_id_dict_sim2nl
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
        # if self.is_working_memory:
        #     for nl_grabbalbe_obj in nl_grabbable_names:
        #         skill_set.append(f'recall location of {nl_grabbalbe_obj}')
        # if self.is_scene_graph:
        #     skill_set.append(f'retrieve information of {nl_grabbalbe_obj}')
        return skill_set