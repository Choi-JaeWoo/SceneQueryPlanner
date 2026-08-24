import os
import sys
import networkx as nx


from core.wm.sg_base import BaseSG
import matplotlib.pyplot as plt
import core.utils.procthor_utils as utils

class ProcThorWM():
    """
    Naive Working Memory.
    """
    def __init__(self, env):
        """
        Initialize the planner with a simulation environment and an LLM-based decision-making agent.
        """
        self.scene_graph = {}
        self.env = env
    
    def update_graph(self, sim_skill_info=None):
        partial_graph = self.env.get_partial_graph()
        agent_id = self.env.agent_id
        if sim_skill_info == None:
            sim_act = None
        else:
            sim_act = sim_skill_info['sim_act']
            sim_obj_info = sim_skill_info['sim_obj_info']
        
        if sim_act == None:
            ### TODO: All room and assets
            init_graph = self.env.get_graph_obs(visibility='initial')
            id2node = {node['id']: node for node in init_graph['nodes']}
            sim_room_infos = utils.obs_all_rooms(init_graph)        # objectType, sg_id
            sim_asset_infos = [(node['objectType'], node['id']) for node in init_graph['nodes'] if node['category'] == 'asset']       # objectType, sg_id
            for sim_asset_info in sim_asset_infos:
                location_info = utils.get_node_location_details(init_graph, sim_asset_info[1])
                room_id = location_info['room_ids'][0]
                sim_room_info = (id2node[room_id]['objectType'], id2node[room_id]['objectId'])
                nl_room_info = self.env.name_id_dict_sim2nl[sim_room_info]
                asset_id = sim_asset_info[1]
                sim_asset_info = (id2node[asset_id]['objectType'], id2node[asset_id]['objectId'])
                nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_asset_info]
                utils.update_working_memory(self.scene_graph, nl_target_obj_info, nl_room_info, None)           
        elif sim_act == 'TeleportFull':
            ### Observations for 1) go to room, or 2) go to object
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            if sim_agent_room_info is None or sim_agent_room_info not in self.env.name_id_dict_sim2nl:
                return
            nl_agent_room_info = self.env.name_id_dict_sim2nl[sim_agent_room_info]

            if sim_obj_info[0] in ['kitchen', 'livingroom', 'bedroom', 'bathroom']:
                sim_room_item_infos = utils.obs_room_items(partial_graph['nodes'], self.env.last_event)
                for sim_target_obj_info in sim_room_item_infos:
                    nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.scene_graph, nl_target_obj_info, nl_agent_room_info, None)
            else:
                sim_location_obj_info = sim_skill_info['sim_obj_info']
                if sim_location_obj_info[0] in self.env.sim_receps:
                    sim_close_obj_infos = utils.obs_close_objs_recep(partial_graph['nodes'], self.env.last_event, self.env.cur_recep_info)
                    nl_location_obj_info = self.env.name_id_dict_sim2nl[sim_location_obj_info]
                    for sim_target_obj_info in sim_close_obj_infos:
                        if sim_target_obj_info not in self.env.name_id_dict_sim2nl:
                            continue
                        nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                        utils.update_working_memory(self.scene_graph, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)
                else:
                    sim_close_obj_infos = utils.obs_close_objs(partial_graph['nodes'], self.env.last_event, self.env.cur_recep_info)
                    if self.env.cur_recep_info not in self.env.name_id_dict_sim2nl:
                        nl_location_obj_info = None
                    else:
                        nl_location_obj_info = self.env.name_id_dict_sim2nl[self.env.cur_recep_info]
                    for sim_target_obj_info in sim_close_obj_infos:
                        if sim_target_obj_info not in self.env.name_id_dict_sim2nl:
                            continue
                        nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                        utils.update_working_memory(self.scene_graph, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)
            sim_grab_obj_infos = utils.obs_agent_grab(self.env.last_event)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.scene_graph, nl_target_obj_info, ('agent', 1), ('agent', 1))    
        elif sim_act == 'PickupObject':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.env.name_id_dict_sim2nl[sim_agent_room_info]

            sim_grab_obj_infos = utils.obs_agent_grab(self.env.last_event)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.scene_graph, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act == 'PutObject':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.env.name_id_dict_sim2nl[sim_agent_room_info]
            nl_location_obj_info = self.env.name_id_dict_sim2nl[self.env.cur_recep_info]
            sim_target_obj_info = sim_skill_info['sim_obj_info']
            nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
            utils.update_working_memory(self.scene_graph, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)
            sim_grab_obj_infos = utils.obs_agent_grab(self.env.last_event)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.scene_graph, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act == 'OpenObject':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.env.name_id_dict_sim2nl[sim_agent_room_info]
            sim_close_obj_infos = utils.obs_close_objs_recep(partial_graph['nodes'], self.env.last_event, self.env.cur_recep_info)
            sim_location_obj_info = sim_skill_info['sim_obj_info']
            nl_location_obj_info = self.env.name_id_dict_sim2nl[sim_location_obj_info]
            for sim_target_obj_info in sim_close_obj_infos:
                if sim_target_obj_info not in self.env.name_id_dict_sim2nl:
                    continue
                nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                utils.update_working_memory(self.scene_graph, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)
            sim_grab_obj_infos = utils.obs_agent_grab(self.env.last_event)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.scene_graph, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act == 'SliceObject':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.env.name_id_dict_sim2nl[sim_agent_room_info]
            sim_close_obj_infos = utils.obs_close_objs(partial_graph['nodes'], self.env.last_event, self.env.cur_recep_info)
            nl_location_obj_info = self.env.name_id_dict_sim2nl[self.env.cur_recep_info]
            for sim_target_obj_info in sim_close_obj_infos:
                if sim_target_obj_info not in self.env.name_id_dict_sim2nl:
                    continue
                nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                utils.update_working_memory(self.scene_graph, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)
            sim_grab_obj_infos = utils.obs_agent_grab(self.env.last_event)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    if sim_target_obj_info not in self.env.name_id_dict_sim2nl:
                        continue
                    nl_target_obj_info = self.env.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.scene_graph, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act in ['CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'DropHandObject']:
            pass
        else:
            raise NotImplementedError()