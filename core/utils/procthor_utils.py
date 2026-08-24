import os
import re
import gzip
import copy
import numpy as np
from tqdm import tqdm
from prior import LazyJsonDataset, DatasetDict
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import textwrap

def load_dataset() -> DatasetDict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "test.jsonl.gz")
    
    with gzip.open(data_path, "r") as f:
        houses = [line for line in tqdm(f, total=1000, desc="Loading test")]
    return DatasetDict(
        test=LazyJsonDataset(data=houses, dataset="procthor-dataset", split="test")
    )

def extract_graph_by_class_names(graph, class_list):
    graph_extract = {}
    graph_extract['nodes'] = [node for node in graph['nodes'] if node['objectType'] in class_list]
    extracted_ids = [node['id'] for node in graph_extract['nodes']]
    graph_extract['edges'] = [edge for edge in graph['edges'] if (edge['from_id'] in extracted_ids and edge['to_id'] in extracted_ids)]
    return graph_extract

def obs_room_items(partial_graph, event):
    room_items = ['armchair', 'bathtub', 'bathtubbasin', 'bed', 'cabinet', 'coffeetable', 'countertop', 'desk', 'diningtable', 'drawer', 'dresser', 'fridge', 'garbagecan', 'handtowelholder', 'laundryhamper', 'ottoman', 'safe', 'shelf', 'sidetable', 'sink', 'sinkbasin', 'sofa', 'stoveburner', 'toilet', 'toiletpaperhanger', 'towelholder', 'tvstand', 'washingmachine', 'chair', 'stool', 'dogbed', 'dumbbell', 'footstool']
    
    inventory_ids = {
        obj['objectId']
        for obj in event.metadata.get('inventoryObjects', [])
    }
    obs_room_items = []
    for node in partial_graph:
        obj_type = node.get('objectType', '')
        obj_id = node.get('objectId')
        parent_receptacles = node.get('parentReceptacles', [])
        if obj_id in inventory_ids:
            continue 
        if obj_type in room_items:
            obs_room_items.append((obj_type, obj_id))
            continue

        if not parent_receptacles or all(pr == 'Floor' for pr in parent_receptacles):
            obs_room_items.append((obj_type, obj_id))

    return obs_room_items

##### Natural language transformation functions
def make_name_id_dict(graph, obj_dict_sim2nl, room_info):
    id2node = {node['objectId']: node for node in graph}
    class_id_dict = defaultdict(list)
    for obj_id, node in id2node.items():
        class_name = node['objectType']
        class_id_dict[class_name].append(obj_id)
        
    # Organize room nodes (use roomType as key)
    for room_id, room_data in room_info.items():
        room_type = room_data["roomType"]
        class_id_dict[room_type].append(room_id)

    transformed_ids = {}
    for class_name, ids in class_id_dict.items():
        ids.sort()
        for index, obj_id in enumerate(ids, start=1):
            transformed_ids[obj_id] = index

    name_id_dict_sim2nl = {}
    name_id_dict_nl2sim = {}
    for obj_id, node in id2node.items():
        class_name = node['objectType']
        name_id_sim = (class_name, obj_id)
        name_id_nl = (obj_dict_sim2nl[class_name], transformed_ids[obj_id])
        
        name_id_dict_sim2nl[name_id_sim] = name_id_nl
        name_id_dict_nl2sim[name_id_nl] = name_id_sim
        
    # Handle room nodes (use roomType as class_name)
    for room_id, room_data in room_info.items():
        class_name = room_data["roomType"]  # e.g., "bedroom"
        name_id_sim = (class_name, room_id)
        nl_name = obj_dict_sim2nl.get(class_name, class_name)
        name_id_nl = (nl_name, transformed_ids[room_id])
        name_id_dict_sim2nl[name_id_sim] = name_id_nl
        name_id_dict_nl2sim[name_id_nl] = name_id_sim

    name_id_dict_sim2nl[("user", "agent")] = ("user", 1)
    name_id_dict_nl2sim[("user", 1)] = ("user", "agent")
    return name_id_dict_sim2nl, name_id_dict_nl2sim

def merge_obs_list(obs_sim_list, name_id_dict_sim2nl):
    obs_nl_summary = defaultdict(list)

    for sim_name, sim_id in obs_sim_list:
        if (sim_name, sim_id) in name_id_dict_sim2nl:
            natural_name, natural_id = name_id_dict_sim2nl[(sim_name, sim_id)]
            obs_nl_summary[natural_name].append(natural_id)
    
    result_str = []
    for name, ids in sorted(obs_nl_summary.items()):
        ids_sorted = sorted(set(ids))
        id_str = ', '.join(str(id) for id in ids_sorted)
        result_str.append(f'{name} ({id_str})')
        # result_str.append(f'{name} {id_str}')
    return ', '.join(result_str)

def filter_ignored_nodes(nodes, exclude):
    ignored_types = {"floor", "window", "wall", "door", "doorway", "doorframe"}
    exclude_ids = set(exclude)
    filtered_nodes = []
    for node in nodes:
        obj_type = node.get("objectType", "").lower()
        obj_id = node.get("objectId", "")
        if obj_type not in ignored_types and obj_id not in exclude_ids:
            node = node.copy() 
            node["objectType"] = obj_type 
            filtered_nodes.append(node)
    return filtered_nodes

def decompose_nl_skill(nl_skill, name_id_dict_sim2nl):
    act_dict_nl2sim = {
        'go to': 'TeleportFull', 'pick up': 'PickupObject', 'put down': 'PutObject',
        'open': 'OpenObject', 'close': 'CloseObject', 'turn on': 'ToggleObjectOn',
        'turn off': 'ToggleObjectOff', 'slice': 'SliceObject', 'drop': 'DropHandObject'
    }

    def get_sim_obj_info(nl_obj_phrase):
        nl_obj_info = split_nl_name_id(nl_obj_phrase)
        if nl_obj_info not in name_id_dict_sim2nl:
            return None, nl_obj_info
        return name_id_dict_sim2nl[nl_obj_info], None

    for act_phrase, sim_act in act_dict_nl2sim.items():
        if act_phrase + ' ' in nl_skill:
            nl_obj_phrase = nl_skill.split(act_phrase + ' ')[1]
            sim_obj_info, missing_obj = get_sim_obj_info(nl_obj_phrase)
            if sim_obj_info is None:
                return {
                    'sim_act': None,
                    'sim_obj_info': missing_obj
                }
            return {
                'sim_act': sim_act,
                'sim_obj_info': sim_obj_info
            }

    return {
        'sim_act': None,
        'sim_obj_info': None
    }
    
def split_nl_name_id(nl_name_id):
    match = re.match(r'^(.+?)\s(\d+)$', nl_name_id)
    if match:
        object_name, index = match.groups()
        index = int(index)
        return (object_name, index)
    else:
        return "Invalid input", -1
    
def check_attributes(graph, obj_id, attribute):
    for obj in graph:
        if obj["objectId"] == obj_id:
            return obj.get(attribute, False)
    return False

def obs_close_objs_recep(partial_graph, event, cur_recep_info, max_distance=2.0):
    obs_result = []
    inventory_ids = {
        obj['objectId']
        for obj in event.metadata.get('inventoryObjects', [])
    }
    _, cur_recep_id = cur_recep_info
    for obj in partial_graph:
        obj_id = obj.get('objectId')
        if obj_id in inventory_ids:
            continue
        obj_type = obj.get('objectType')
        parent_receptacles = obj.get('parentReceptacles') or []
        if cur_recep_id in parent_receptacles:
            obs_result.append((obj_type, obj_id))
            continue
        distance = obj.get('distance', float('inf'))
        if distance <= max_distance:
            obs_result.append((obj_type, obj_id))
    return obs_result

def obs_close_objs(partial_graph, event, cur_recep_info, max_distance=2.0):
    obs_result = []
    inventory_ids = {
        obj['objectId']
        for obj in event.metadata.get('inventoryObjects', [])
    }
    for obj in partial_graph:
        if obj.get('objectId') in inventory_ids:
            continue  
        if obj.get('distance', float('inf')) <= max_distance:
            obs_result.append((obj['objectType'], obj['objectId']))
    return obs_result

def obs_agent_grab(event):
    inventory = event.metadata.get('inventoryObjects', [])
    if not inventory:
        return None
    obj = inventory[0] 
    return [(obj['objectType'].lower(), obj['objectId'])]

def obs_partial_objs(partial_graph):
    partial_objs = []
    for node in partial_graph:
        partial_objs.append((node['objectType'], node['objectId']))
    return partial_objs

def sort_with_same_similarity(sorted_ic_ex_encode_list):
    final_list = []
    current_similarity = None
    same_similarity_list = []
    
    for experience in sorted_ic_ex_encode_list:
        similarity = experience["similarity"]

        if current_similarity is None:
            current_similarity = similarity

        # Same similarity as the current group
        if similarity == current_similarity:
            same_similarity_list.append(experience)
        else:
            # When similarity changes, process the previous group and reset
            final_list.extend(process_same_similarity_list(same_similarity_list))
            current_similarity = similarity
            same_similarity_list = [experience]
    # Process the last similarity group
    if same_similarity_list:
        final_list.extend(process_same_similarity_list(same_similarity_list))
    return final_list

def process_same_similarity_list(experience_list, select_next='expand'):
    result_list = []
    success_list = []
    failure_list = []
    expand_list = []
    etc_list = []
    for exp in experience_list:
        if exp["text_trajectory"].endswith('done'):
            success_list.append(exp)
        elif exp["text_trajectory"].endswith('failure'):
            failure_list.append(exp)
        elif 'Expand:' in exp["text_trajectory"]:
            expand_list.append(exp)
        else:
            etc_list.append(exp)
    # Add expand, success, failure in alternation
    while expand_list or success_list or failure_list:
        if select_next == "expand" and expand_list:
            result_list.append(expand_list.pop(0))
            select_next = "success"
        elif select_next == "success" and success_list:
            result_list.append(success_list.pop(0))
            select_next = "failure"
        elif select_next == "failure" and failure_list:
            result_list.append(failure_list.pop(0))
            select_next = "expand"
        # If the selected list is empty, move on to the next one
        else:
            if select_next == "expand":
                select_next = "success" if success_list else "failure"
            elif select_next == "success":
                select_next = "failure" if failure_list else "expand"
            elif select_next == "failure":
                select_next = "expand" if expand_list else "success"
    # Add any remaining success/failure items (when alternation could not pick them all)
    result_list.extend(etc_list)
    return result_list


##### Evaluation
def check_goal_condition(task, graph, init_event, cleaned_objects, cooled_objects, heated_objects, filled_coffee_objects):
    mode = task['mode']
    task_goal = task['task_goal']
    
    if mode == "PickAndPlaceSingleTask":
        subgoal_success_rate = check_goal_single(task_goal, graph)
    elif mode == "PickAndPlaceMultipleTask":
        subgoal_success_rate = check_goal_multiple(task_goal, graph)
    elif mode == "PickAndPlaceImplicitTask":
        subgoal_success_rate = check_goal_implicit(task_goal, graph, init_event)
    elif mode == "LookAtObjInLightTask":
        subgoal_success_rate = check_goal_light(task_goal, graph)
    elif mode == "PickHeatThenPlaceInRecepTask":
        subgoal_success_rate = check_goal_heat(task_goal, graph, heated_objects)
    elif mode == "PickCoolThenPlaceInRecepTask":
        subgoal_success_rate = check_goal_cool(task_goal, graph, cooled_objects)
    elif mode == "ToggleOffTask":
        subgoal_success_rate = check_goal_toggle(task_goal, graph, init_event)
    elif mode == "PickCleanThenPlaceInRecepTask":
        subgoal_success_rate = check_goal_clean(task_goal, graph, cleaned_objects)
    elif mode == "PickAndStackTask":
        subgoal_success_rate = check_goal_stack(task_goal, graph)
    elif mode == "CookTask":
        subgoal_success_rate = check_goal_cook(task_goal, graph, filled_coffee_objects, heated_objects)
    
    return subgoal_success_rate

def check_goal_single(task_goal, controller):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    sliced_required = set()
    for key in task_goal:
        if key.startswith('isSliced_'):
            sliced_obj = key.split('_')[1].lower()
            if any(
                sliced_obj in obj['objectType'].lower() and 'sliced' in obj['objectType'].lower()
                for obj in objects
            ):
                s += 1
                sliced_required.add(sliced_obj)

    for key in task_goal:
        if key.startswith('isSliced_'):
            continue 
        parts = key.split('_')
        if parts[0] in ['on', 'in'] and len(parts) == 3:
            obj_name, recep_name = parts[1].lower(), parts[2].lower()
            obj_check_name = obj_name + 'sliced' if obj_name in sliced_required else obj_name
            receptacles = get_objects_with_name_and_prop(recep_name, 'receptacle', objects)
            objs = get_objects_with_name_and_prop(obj_check_name, 'pickupable', objects)
            obj_ids = {obj['objectId'] for obj in objs}
            if any(obj_ids & set(recep.get('receptacleObjectIds', [])) for recep in receptacles):
                s += 1
    return s / ts if ts > 0 else 0.0

def check_goal_multiple(task_goal, controller):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    sliced_required = set()
    for key in task_goal:
        if key.startswith('isSliced_'):
            sliced_obj = key.split('_')[1].lower()
            if any(
                sliced_obj in obj['objectType'].lower() and 'sliced' in obj['objectType'].lower()
                for obj in objects
            ):
                s += 1
                sliced_required.add(sliced_obj)

    for key in task_goal:
        if key.startswith('isSliced_'):
            continue
        parts = key.split('_')
        if parts[0] in ['on', 'in'] and len(parts) == 3:
            obj_name, recep_name = parts[1].lower(), parts[2].lower()
            obj_check_name = obj_name + 'sliced' if obj_name in sliced_required else obj_name
            receptacles = get_objects_with_name_and_prop(recep_name, 'receptacle', objects)
            objs = get_objects_with_name_and_prop(obj_check_name, 'pickupable', objects)
            obj_ids = {obj['objectId'] for obj in objs}
            if any(obj_ids & set(recep.get('receptacleObjectIds', [])) for recep in receptacles):
                s += 1
    return s / ts if ts > 0 else 0.0

def check_goal_implicit(task_goal, controller, init_event):
    s = 0
    objects_now = controller.last_event.metadata["objects"]
    objects_init = init_event.metadata["objects"]
    now_lookup = {obj['objectId']: obj for obj in objects_now}
    init_lookup = {obj['objectId']: obj for obj in objects_init}
    # Extract source and target receptacle types
    source_types = {k.split("empty_")[1].lower() for k in task_goal if k.startswith("empty_")}
    target_types = {k.split("moveTo_")[1].lower() for k in task_goal if k.startswith("moveTo_")}
    # Get object IDs that were in source receptacles and are pickupable
    init_source_obj_ids = {
        oid
        for obj in objects_init
        if obj['objectType'].lower() in source_types
        for oid in obj.get('receptacleObjectIds', [])
        if init_lookup.get(oid, {}).get('pickupable', True)
    }
    ts = len(init_source_obj_ids) * 2  # each object is evaluated for empty and moveTo
    # Step 1: Check if each object has been removed from any source receptacle
    source_recep_map = {
        obj['objectId']: obj
        for obj in objects_now if obj['objectType'].lower() in source_types
    }
    for oid in init_source_obj_ids:
        if not any(oid in recep.get('receptacleObjectIds', []) for recep in source_recep_map.values()):
            s += 1  # successfully removed from source
    # Step 2: Check if each object is now in any target receptacle
    target_recep_map = {
        obj['objectId']: obj
        for obj in objects_now if obj['objectType'].lower() in target_types
    }
    for oid in init_source_obj_ids:
        if any(oid in recep.get('receptacleObjectIds', []) for recep in target_recep_map.values()):
            s += 1  # successfully placed in target
    return s / ts if ts > 0 else 0.0

def check_goal_light(task_goal, controller):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    inventory = controller.last_event.metadata.get("inventoryObjects", [])
    obj_by_type = defaultdict(list)
    for obj in objects:
        obj_by_type[obj['objectType'].lower()].append(obj)
    inventory_types = {obj['objectType'].lower() for obj in inventory}
    for goal_key in task_goal:
        cond, target = goal_key.split('_', 1)
        target = target.lower()
        objs = obj_by_type.get(target, [])
        if cond == 'isToggled':
            if any(obj['isToggled'] for obj in objs):
                s += 1
        elif cond == 'visible':
            if any(obj['isToggled'] and obj['visible'] for obj in objs):
                s += 1
        elif cond == 'isPickedUp':
            if target in inventory_types:
                s += 1
    return s / ts if ts > 0 else 0.0

def check_goal_heat(task_goal, controller, heated_objects):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    obj_by_type = defaultdict(list)
    for obj in objects:
        obj_by_type[obj['objectType'].lower()].append(obj)
    sliced_required = set()
    for key in task_goal:
        if key.startswith("isSliced_"):
            obj_type = key.split('_')[1].lower()
            for obj in obj_by_type.get(obj_type + 'sliced', []):
                sliced_required.add(obj_type)
                s += 1
                break
    for key in task_goal:
        if key.startswith("isSliced_"):
            continue
        parts = key.split('_')
        cond, *args = parts
        # temperatureHot_<obj>
        if cond == 'temperatureHot':
            obj_type = args[0].lower()
            if any(obj_type in obj_id.lower() for obj_id in heated_objects):
                s += 1
        # on_<obj>_<recep> or in_<obj>_<recep>
        elif cond in ['on', 'in'] and len(args) == 2:
            obj_type, recep_type = args[0].lower(), args[1].lower()
            obj_ids = set()
            # For bread, include sliced as well
            if obj_type == 'bread':
                obj_ids |= {o['objectId'] for o in obj_by_type.get('bread', [])}
                obj_ids |= {o['objectId'] for o in obj_by_type.get('breadsliced', [])}
            else:
                obj_check = obj_type + 'sliced' if obj_type in sliced_required else obj_type
                obj_ids = {o['objectId'] for o in obj_by_type.get(obj_check, [])}
            for recep in obj_by_type.get(recep_type, []):
                if obj_ids & set(recep.get('receptacleObjectIds', [])):
                    s += 1
                    break
        # on_temperatureHot_<obj>_<recep> or in_temperatureHot_<obj>_<recep>
        elif cond in ['on', 'in'] and args[0] == 'temperatureHot':
            obj_type, recep_type = args[1].lower(), args[2].lower()
            for recep in obj_by_type.get(recep_type, []):
                if heated_objects & set(recep.get('receptacleObjectIds', [])):
                    s += 1
                    break
    return s / ts if ts > 0 else 0.0

def check_goal_cool(task_goal, controller, cooled_objects):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    obj_by_type = defaultdict(list)
    for obj in objects:
        obj_by_type[obj['objectType'].lower()].append(obj)
    sliced_required = set()
    for key in task_goal:
        if key.startswith("isSliced_"):
            obj_type = key.split("_")[1].lower()
            for obj in obj_by_type.get(obj_type + "sliced", []):
                sliced_required.add(obj_type)
                s += 1
                break
    for key in task_goal:
        if key.startswith("isSliced_"):
            continue
        parts = key.split('_')
        cond, *args = parts
        # temperatureCold_<obj>
        if cond == 'temperatureCold':
            obj_type = args[0].lower()
            if any(obj_type in obj_id.lower() for obj_id in cooled_objects):
                s += 1
        # on_<obj>_<recep> or in_<obj>_<recep>
        elif cond in ['on', 'in'] and len(args) == 2:
            obj_type, recep_type = args[0].lower(), args[1].lower()
            obj_check = obj_type + 'sliced' if obj_type in sliced_required else obj_type
            obj_ids = {obj['objectId'] for obj in obj_by_type.get(obj_check, [])}
            for recep in obj_by_type.get(recep_type, []):
                if obj_ids & set(recep.get('receptacleObjectIds', [])):
                    s += 1
                    break
        # on_temperatureCold_<obj>_<recep> or in_temperatureCold_<obj>_<recep>
        elif cond in ['on', 'in'] and args[0] == 'temperatureCold':
            obj_type, recep_type = args[1].lower(), args[2].lower()
            for recep in obj_by_type.get(recep_type, []):
                if cooled_objects & set(recep.get('receptacleObjectIds', [])):
                    s += 1
                    break
    return s / ts if ts > 0 else 0.0

def check_goal_toggle(task_goal, controller, init_event):
    s = 0
    ts = 0
    objects_now = controller.last_event.metadata["objects"]
    objects_init = init_event.metadata["objects"]
    now_by_type = defaultdict(list)
    init_by_type = defaultdict(list)
    for obj in objects_now:
        now_by_type[obj['objectType'].lower()].append(obj)
    for obj in objects_init:
        init_by_type[obj['objectType'].lower()].append(obj)
    for key in task_goal:
        if key.startswith("toggleOff_"):
            target_type = key.split("_", 1)[1].lower()
            # Filter objects toggled on in the initial state
            init_objs = init_by_type.get(target_type, [])
            toggled_on_ids = {
                obj['objectId'] for obj in init_objs if obj.get('isToggled', False)
            }
            ts += len(toggled_on_ids)
            # Check whether those objects are toggled off in the current state
            now_objs = now_by_type.get(target_type, [])
            now_lookup = {obj['objectId']: obj for obj in now_objs}
            for obj_id in toggled_on_ids:
                if obj_id in now_lookup and not now_lookup[obj_id].get('isToggled', True):
                    s += 1
    return s / ts if ts > 0 else 0.0

def check_goal_clean(task_goal, controller, cleaned_objects):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    # objectId -> object dict
    obj_lookup = {obj['objectId']: obj for obj in objects}
    # objectType -> set of objectIds
    obj_ids_by_type = defaultdict(set)
    for obj in objects:
        obj_ids_by_type[obj['objectType'].lower()].add(obj['objectId'])
    # receptacleType -> set of objectIds inside
    recep_obj_ids_by_type = defaultdict(set)
    for obj in objects:
        if obj.get('receptacle', False):
            recep_type = obj['objectType'].lower()
            recep_obj_ids_by_type[recep_type].update(obj.get('receptacleObjectIds', []))
    for key in task_goal:
        parts = key.split('_')
        # Case 1: isClean_<obj>
        if key.startswith("isClean_"):
            obj_type = parts[1].lower()
            if any(obj_id for obj_id in cleaned_objects if obj_lookup.get(obj_id, {}).get("objectType", "").lower() == obj_type):
                s += 1
        # Case 2: on_<obj>_<recep> or in_<obj>_<recep>
        elif parts[0] in ['in', 'on'] and len(parts) == 3:
            _, obj_type, recep_type = parts
            obj_ids = obj_ids_by_type.get(obj_type.lower(), set())
            recep_ids = recep_obj_ids_by_type.get(recep_type.lower(), set())
            if obj_ids & recep_ids:
                s += 1
        # Case 3: on_isClean_<obj>_<recep> or in_isClean_<obj>_<recep>
        elif parts[0] in ['in', 'on'] and parts[1] == 'isClean':
            _, _, obj_type, recep_type = parts
            recep_ids = recep_obj_ids_by_type.get(recep_type.lower(), set())
            clean_obj_ids = {
                obj_id for obj_id in cleaned_objects
                if obj_lookup.get(obj_id, {}).get("objectType", "").lower() == obj_type.lower()
            }
            if clean_obj_ids & recep_ids:
                s += 1
    return s / ts if ts > 0 else 0.0
    
from collections import defaultdict

def check_goal_stack(task_goal, controller):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    inventory_objs = controller.last_event.metadata.get("inventoryObjects", [])
    obj_by_type = defaultdict(list)
    obj_by_id = {}
    for obj in objects:
        obj_type = obj["objectType"].lower()
        obj_by_type[obj_type].append(obj)
        obj_by_id[obj["objectId"]] = obj
    # Precompute sliced_required types
    sliced_required = {
        key.split("_")[1].lower()
        for key in task_goal
        if key.startswith("isSliced_") and obj_by_type.get(key.split("_")[1].lower() + "sliced")
    }
    s += len(sliced_required)  # reward 1 for each valid sliced goal
    for key in task_goal:
        if key.startswith("isSliced_"):
            continue
        parts = key.split("_")
        cond = parts[0]
        if cond in ("on", "in") and len(parts) == 3:
            obj_name, recep_name = parts[1].lower(), parts[2].lower()
            obj_check = obj_name + "sliced" if obj_name in sliced_required else obj_name
            obj_ids = {obj["objectId"] for obj in obj_by_type.get(obj_check, [])}
            for recep in obj_by_type.get(recep_name, []):
                if obj_ids & set(recep.get("receptacleObjectIds", [])):
                    s += 1
                    break
        elif cond == "stack" and len(parts) == 4:
            inner_name, outer_name, base_name = map(str.lower, parts[1:])
            inner_name = inner_name + "sliced" if inner_name in sliced_required else inner_name
            inner_ids = {obj["objectId"] for obj in obj_by_type.get(inner_name, [])}
            outer_objs = obj_by_type.get(outer_name, [])
            base_objs = obj_by_type.get(base_name, [])
            for outer in outer_objs:
                if not (inner_ids & set(outer.get("receptacleObjectIds", []))):
                    continue
                outer_id = outer["objectId"]
                if any(outer_id in base.get("receptacleObjectIds", []) for base in base_objs):
                    s += 1
                    break
    return s / ts if ts > 0 else 0.0

def check_goal_cook(task_goal, controller, filled_coffee_objects, heated_objects):
    s = 0
    ts = len(task_goal)
    objects = controller.last_event.metadata["objects"]
    obj_by_type = defaultdict(list)
    obj_by_id = {}
    for obj in objects:
        obj_type = obj['objectType'].lower()
        obj_by_type[obj_type].append(obj)
        obj_by_id[obj['objectId']] = obj
    for key in task_goal:
        parts = key.split('_')
        prefix = parts[0]
        if prefix == 'isCooked':
            obj_type = parts[1].lower()
            if obj_type in {'potato', 'egg'}:
                # Check if heated_objects contains an objectId matching this obj_type
                if any(obj_type in obj_id.lower() for obj_id in heated_objects):
                    s += 1
            else:
                s += any(obj['isCooked'] for obj in obj_by_type.get(obj_type, []))
        elif prefix in {'on', 'in'} and len(parts) == 3:
            obj_type = parts[1].lower()
            recep_type = parts[2].lower()
            obj_ids = {obj['objectId'] for obj in obj_by_type.get(obj_type, [])}
            if any(obj_ids & set(recep.get('receptacleObjectIds', [])) for recep in obj_by_type.get(recep_type, [])):
                s += 1
        elif prefix == 'fillLiquid' and len(parts) == 3:
            container_type = parts[2].lower()
            if any(container_type in obj_id.lower() for obj_id in filled_coffee_objects):
                s += 1
    return s / ts if ts > 0 else 0.0

def get_objects_with_name_and_prop(name, prop, metadata):
    return [obj for obj in metadata
            if name == obj['objectType'].lower() and obj[prop]]
    
def get_distance(pos1, pos2):
    return np.linalg.norm(np.array([pos1[k] for k in ["x", "y", "z"]]) - 
                          np.array([pos2[k] for k in ["x", "y", "z"]]))
    
def save_vis_log(out_dir, vis_log, env_id, mode, nl_inst):
    base_path = out_dir
    img_list = [vis['images'][0] for vis in vis_log]
    text_list = [vis['action'] for vis in vis_log]
    file_path = os.path.join(base_path, f'{env_id}_{mode}.png')
    save_images_in_grid(img_list, text_list, nl_inst, file_path)
    
def save_images_in_grid(img_list, text_list, title, file_name, grid_width=7, title_height=70, font_path="UbuntuMono-B.ttf", font_size=45):
    processed_imgs = [add_text_to_np_img(img, text) for img, text in zip(img_list, text_list)]
    image_width, image_height = processed_imgs[0].size
    grid_height = len(processed_imgs) // grid_width + (1 if len(processed_imgs) % grid_width else 0)

    total_height = grid_height * image_height + title_height
    total_width = grid_width * image_width
    grid_image = Image.new('RGB', (total_width, total_height), 'white')

    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype(font_path, font_size)
    title_lines = textwrap.wrap(title, width=110)
    # y_start = 10 if len(title_lines) > 1 else 35
    y_start = 10 if len(title_lines) > 1 else 15
    draw.multiline_text((10, y_start), '\n'.join(title_lines), font=font, fill='black')
    
    y_offset = title_height
    for idx, image in enumerate(processed_imgs):
        x_offset = (idx % grid_width) * image_width
        if idx % grid_width == 0 and idx != 0:
            y_offset += image_height
        grid_image.paste(image, (x_offset, y_offset))
    grid_image.save(file_name, 'PNG')
    
def add_text_to_np_img(np_img, text, font_path="UbuntuMono-B.ttf", font_size=35, padding=10):
    img = Image.fromarray(np_img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=font_size)
    
    # Compute line wrapping to fit the image width
    image_width = img.width - 2 * padding  # Image width accounting for left/right padding
    text_lines = []
    
    # Wrap text to an appropriate line length
    words = text.split()
    line = ""
    for word in words:
        # Tentatively add the word and measure the current line width
        test_line = f"{line} {word}".strip()
        # Measure text size with getbbox
        line_bbox = font.getbbox(test_line)
        line_width = line_bbox[2] - line_bbox[0]  # Text width
        
        # Break the line if it exceeds the image width
        if line_width <= image_width:
            line = test_line
        else:
            text_lines.append(line)  # If it overflows, save the current line
            line = word              # and start next line
        
    # Add the last line
    if line:
        text_lines.append(line)

    # Set the y position of the first text line
    y_text = padding
    
    # Draw each line of text
    for line in text_lines:
        draw.text((padding, y_text), line, font=font, fill=(255, 255, 255, 255))
        y_text += font_size + 5  # Add spacing between lines
    
    return img


####### Naive WM
def recall_working_memory(working_memory, target_obj):
    receptacle_classes = [
        'armchair', 'bathtub', 'bathtub basin', 'bed', 'box', 'cabinet', 'coffee table', 'countertop',
        'desk', 'dining table', 'drawer', 'dresser', 'fridge', 'garbage can', 'hand towel holder',
        'laundry hamper', 'ottoman', 'safe', 'shelf', 'side table', 'sink', 'sink basin', 'sofa',
        'stove burner', 'toilet', 'toilet paper hanger', 'towel holder', 'TV stand', 'washing machine',
        'chair', 'stool'
    ]
    if target_obj in working_memory:
        location_infos = working_memory[target_obj]
        messages = []
        for location_info in location_infos:
            target_obj_id = location_info['id']
            nl_room = location_info['room']
            nl_location_obj = location_info['location_obj']
            if 'agent' in nl_room:
                message = f'You are holding {target_obj} {target_obj_id}.'
            else:
                if nl_location_obj:
                    if target_obj in receptacle_classes:
                        message = f'You saw {target_obj} {target_obj_id} in {nl_room}.'
                    else:
                        message = f'You saw {target_obj} {target_obj_id} near {nl_location_obj} in {nl_room}.'
                else:
                    message = f'You saw {target_obj} {target_obj_id} in {nl_room}.'
            messages.append(message)
        obs_text = ' '.join(messages)
    else:
        obs_text = f'You have not seen {target_obj} before.'
    return obs_text

def update_working_memory(working_memory, nl_target_obj_info, nl_room_info, nl_location_obj_info):
    ### working memory: {'class_name': [{'id': , 'room': , 'close_obj': }]}
    if nl_room_info:
        nl_room = f'{nl_room_info[0]} {nl_room_info[1]}'
    else:
        nl_room = None
    if nl_location_obj_info:
        nl_location_obj = f'{nl_location_obj_info[0]} {nl_location_obj_info[1]}'
    else:
        nl_location_obj = None    

    nl_target_obj_name, nl_target_obj_id = nl_target_obj_info[0], nl_target_obj_info[1] 

    if nl_target_obj_name not in working_memory:
        working_memory[nl_target_obj_name] = []
    updated = False
    for entry in working_memory[nl_target_obj_name]:
        if entry['id'] == nl_target_obj_id:
            entry['room'] = nl_room
            entry['location_obj'] = nl_location_obj
            updated = True
    if not updated:
        working_memory[nl_target_obj_name].append({'id': nl_target_obj_id, 'room': nl_room, 'location_obj': nl_location_obj})
        
def obs_all_rooms(graph):
    return [(node['objectType'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms'] # (class name, id)

def get_node_location_details(graph, node_id):
    id2node = {node['id']: node for node in graph['nodes']}
    edges_from_node = find_edges_connected_to_node(graph, node_id)['edges_from_node']
    room_node_ids, in_receptacle_ids, on_receptacle_ids = [], [], []

    for edge in edges_from_node:
        target_node_id = edge['to_id']
        if edge['relation_type'] == 'INSIDE':
            if id2node[target_node_id]['category'] == 'Rooms':
                room_node_ids.append(target_node_id)
            else:
                in_receptacle_ids.append(target_node_id)
        elif edge['relation_type'] == 'ON':
            on_receptacle_ids.append(target_node_id)
    location_details = {
        'room_ids': room_node_ids,
        'in_receptacle_ids': in_receptacle_ids,
        'on_receptacle_ids': on_receptacle_ids
    }
    return location_details

def find_edges_connected_to_node(graph, node_id):
    edges_from_node = [edge for edge in graph['edges'] if edge['from_id']==node_id]
    edges_to_node = [edge for edge in graph['edges'] if edge['to_id']==node_id]
    return {'edges_from_node': edges_from_node, 
            'edges_to_node': edges_to_node}
    
def obs_agent_room(graph, agent_id):
    agent_room_ids = get_node_location_details(graph, agent_id)['room_ids']
    id2node = {node['id']: node for node in graph['nodes']}
    if len(agent_room_ids) == 1:
        agent_room_node = id2node[agent_room_ids[0]]
        return (agent_room_node['objectType'], agent_room_node['objectId'])
    else:
        return None
    
def sim_id_to_sg_id(graph, sim_id):
    for node in graph['nodes']:
        if node.get('objectId') == sim_id:
            return node['id']
    raise ValueError(f"objectId '{sim_id}' not found in graph.")

def sg_id_to_sim_id(graph, sg_id):
    for node in graph['nodes']:
        if node.get('id') == sg_id:
            return node.get('objectId')
    raise ValueError(f"id '{sg_id}' not found in graph.")