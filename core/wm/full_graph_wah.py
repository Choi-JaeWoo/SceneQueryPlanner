import os
import sys
import networkx as nx


from core.utils.wah_utils import get_node_location_details
from core.wm.sg_base import BaseSG
import matplotlib.pyplot as plt
import core.utils.wah_utils as wah_utils
import math

def euclidean(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))



class WahFullGraph():
    def __init__(self, cfg, full_graph, name_id_dict_sim2nl, name_id_dict_nl2sim, task_data=None):
        """
        Initialize the planner with a simulation environment and an LLM-based decision-making agent.
        """
        self.cfg = cfg
        self.name_id_dict_sim2nl = name_id_dict_sim2nl
        self.name_id_dict_nl2sim = name_id_dict_nl2sim
    
        self.scene_graph = self.convert_nx_graph(full_graph, task_data)
    
    def convert_nx_graph(self, sim_graph, task_data=None):
        G = nx.MultiDiGraph()
        name_id_dict_sim2nl = self.name_id_dict_sim2nl
        id2node = {node['id']: node for node in sim_graph.get("nodes", [])}
        for node in sim_graph.get("nodes", []):
            name_id_sim = (node["class_name"], node["id"])
            name_id_nl = name_id_dict_sim2nl[name_id_sim]
            node_key = f"{name_id_nl[0]} {name_id_nl[1]}"
            category = node.get("category", "")
            if category == "Rooms":
                node_type = "room"
            elif category in ["Furniture", "Appliance", "Container", 'Electronics', "Appliances"]:
                node_type = "asset"
            elif category:
                node_type = "object"
            else:
                node_type = "unknown"

            sg_node = {
                'id': f"{name_id_nl[0]} {name_id_nl[1]}",
                'name_id_sim': name_id_sim,
                'name_id_nl': name_id_nl,
                'node_type': node_type,
                'prefab_name': node.get("prefab_name", ""),
                'obj_transform': node.get("obj_transform", ""),
                'bounding_box': node.get("bounding_box", ""),
                'properties': node.get("properties", ""),
                'states': node.get("states", ""),
            }
            
            if not node_type == 'room': #node['class_name'] == 'character':
                room_id = get_node_location_details(sim_graph, node['id'])['room_ids'][0]
                room_name_id_sim = (id2node[room_id]["class_name"], room_id)
                room_name_id_nl = name_id_dict_sim2nl[room_name_id_sim]
                sg_node['location'] = f"{room_name_id_nl[0]} {room_name_id_nl[1]}"

            G.add_node(node_key, **sg_node)
        
        for edge in sim_graph.get("edges", []):
            head_name_id_sim = (id2node[edge["from_id"]]["class_name"], edge["from_id"])
            tail_name_id_sim = (id2node[edge["to_id"]]["class_name"], edge["to_id"])

            head_name_id_nl = name_id_dict_sim2nl[head_name_id_sim]
            tail_name_id_nl = name_id_dict_sim2nl[tail_name_id_sim]
            
            sg_edge = {
                'relation': edge.get("relation_type", ""),
                'head_name_id_nl': head_name_id_nl,
                'tail_name_id_nl': tail_name_id_nl,
                'head_name_id_sim': head_name_id_sim,
                'tail_name_id_sim': tail_name_id_sim,
            }

            head_key = f"{head_name_id_nl[0]} {head_name_id_nl[1]}"
            tail_key = f"{tail_name_id_nl[0]} {tail_name_id_nl[1]}"

            G.add_edge(head_key, tail_key, **sg_edge)
        
        
        # 3. Connect unconnected objects → nearest asset
        object_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "object"]
        asset_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "asset"]

        for obj in object_nodes:
            # find the nearest asset
            obj_pos = G.nodes[obj].get("obj_transform", {}).get("position", None)
            if not obj_pos:
                continue

            min_dist = float('inf')
            closest_asset = None
            
            for asset in asset_nodes:
                asset_pos = G.nodes[asset].get("obj_transform", {}).get("position", None)
                if not asset_pos:
                    continue
                dist = euclidean(obj_pos, asset_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_asset = asset
            # add edge
            if closest_asset:
                # skip if an obj → asset or asset → obj relation already exists
                already_connected = any(
                    (obj == u and closest_asset == v) or (obj == v and closest_asset == u)
                    for u, v in G.edges()
                )
                if not already_connected:
                    G.add_edge(obj, closest_asset, relation="inferred_attachment")
        return G
    
    def collapse_graph(self):
        nodes_to_keep = [n for n, attr in self.scene_graph.nodes(data=True) if attr.get("node_type") == "room" or n == 'user 1']
        subgraph = self.scene_graph.subgraph(nodes_to_keep).copy()
        self.cur_graph = subgraph
        
        sayplan_dict = self.scene_graph_to_sayplan_dict()
        return sayplan_dict

    def expand_node(self, node_id):
        if node_id not in self.cur_graph.nodes:
            print(f"Node {node_id} not found in the current graph.")
            return 
        
        base_node_data = self.scene_graph.nodes[node_id]
        base_type = base_node_data.get("node_type", None)

        if base_type != "room":
            print(f"expand_node only works for node_type='room', but got '{base_type}'")
            return

        self.cur_graph.add_node(node_id, **base_node_data)
        
        # 2. Traverse room → asset
        room_neighbors = set(self.scene_graph.successors(node_id)) | set(self.scene_graph.predecessors(node_id))
        for asset_id in room_neighbors:
            asset_data = self.scene_graph.nodes[asset_id]
            # if asset_data.get("node_type") != "asset":
            #     continue

            # 2-1. Add asset
            self.cur_graph.add_node(asset_id, **asset_data)
            self._copy_edges_between(self.scene_graph, self.cur_graph, node_id, asset_id)

            # 2-2. Traverse asset → object
            asset_neighbors = set(self.scene_graph.successors(asset_id)) | set(self.scene_graph.predecessors(asset_id))
            for obj_id in asset_neighbors:
                obj_data = self.scene_graph.nodes[obj_id]
                if obj_data.get("node_type") != "object":
                    continue

                self.cur_graph.add_node(obj_id, **obj_data)
                self._copy_edges_between(self.scene_graph, self.cur_graph, asset_id, obj_id)

        sayplan_dict = self.scene_graph_to_sayplan_dict()
        return sayplan_dict


        # neighbors = set(self.scene_graph.successors(node_id)) | set(self.scene_graph.predecessors(node_id))
        # for neighbor_id in neighbors:
            
        #     neighbor_data = self.scene_graph.nodes[neighbor_id]
        #     neighbor_type = neighbor_data.get("node_type", "")

        #     # condition 1: room → asset
        #     if base_type == "room" and neighbor_type == "asset":
        #         self.cur_graph.add_node(neighbor_id, **neighbor_data)
        #         self._copy_edges_between(self.scene_graph, self.cur_graph, node_id, neighbor_id)

        #     # condition 2: asset → object
        #     elif base_type == "asset" and neighbor_type == "object":
        #         self.cur_graph.add_node(neighbor_id, **neighbor_data)
        #         self._copy_edges_between(self.scene_graph, self.cur_graph, node_id, neighbor_id)
        
        # sayplan_dict = self.scene_graph_to_sayplan_dict()
        # return sayplan_dict
        

        
    def contract_node(self, node_id):
        if node_id not in self.cur_graph.nodes:
            print(f"Node {node_id} not found in current graph.")
            return

        base_node_data = self.cur_graph.nodes[node_id]
        base_type = base_node_data.get("node_type", None)

        if base_type != "room":
            print(f"contract_node only works for node_type='room', but got '{base_type}'")
            return

        # 1. Traverse room → asset
        room_neighbors = set(self.cur_graph.successors(node_id)) | set(self.cur_graph.predecessors(node_id))
        for asset_id in list(room_neighbors):  # defensive copy
            if asset_id not in self.cur_graph:
                continue
            asset_data = self.cur_graph.nodes[asset_id]
            if asset_data.get("node_type") != "asset":
                continue

            # 2. Traverse and remove asset → object
            asset_neighbors = set(self.cur_graph.successors(asset_id)) | set(self.cur_graph.predecessors(asset_id))
            for obj_id in list(asset_neighbors):
                if obj_id in self.cur_graph and self.cur_graph.nodes[obj_id].get("node_type") == "object":
                    self.cur_graph.remove_node(obj_id)

            # 3. Remove asset
            self.cur_graph.remove_node(asset_id)

        sayplan_dict = self.scene_graph_to_sayplan_dict()
        return sayplan_dict

        # # Traverse neighbors based on current cur_graph (both directions)
        # neighbors = set(self.cur_graph.successors(node_id)) | set(self.cur_graph.predecessors(node_id))

        # for neighbor_id in neighbors:
        #     if neighbor_id not in self.cur_graph.nodes:
        #         continue  # guard against edge existing without the node

        #     neighbor_type = self.cur_graph.nodes[neighbor_id].get("node_type", "")

        #     # condition 1: room → asset → remove
        #     if base_type == "room" and neighbor_type == "asset":
        #         self.cur_graph.remove_node(neighbor_id)

        #     # condition 2: asset → object → remove
        #     elif base_type == "asset" and neighbor_type == "object":
        #         self.cur_graph.remove_node(neighbor_id)

        # sayplan_dict = self.scene_graph_to_sayplan_dict()
        # return sayplan_dict
    


    def scene_graph_to_sayplan_dict(self):
        sg = self.cur_graph
        sayplan_graph = {"nodes": {"agent": [], "room": [], "asset": [], "object": []}, "links": []}
        relevant_keys = {"id", "states", "location"}
        for node_key, attr in sg.nodes(data=True):
            node_type = attr.get("node_type", "unknown")
            node_id = node_key

            filtered_attr = {k: v for k, v in attr.items() if k in relevant_keys}

            # Agent
            if node_id == "user 1":
                # Assume agent is inside a room — find linked room via edge or use position overlap
                sayplan_graph["nodes"]["agent"].append(filtered_attr)
                
            # Room
            elif node_type == "room":
                sayplan_graph["nodes"]["room"].append(filtered_attr)
            
            # Asset
            elif node_type == "asset":
                sayplan_graph["nodes"]["asset"].append(filtered_attr)
                

            # Object
            elif node_type == "object":
                sayplan_graph["nodes"]["object"].append(filtered_attr)
                
            # Create links (connect other types linked to room via edges)
        for u, v, edge_attr in sg.edges(data=True):
            u_type = sg.nodes[u].get("node_type")
            v_type = sg.nodes[v].get("node_type")
            # # add link only when agent/asset/object is connected to room
            # if (u_type == "room" and v_type in ["agent", "asset", "object"]):
            #     sayplan_graph["links"].append(f"{u}↔{v}")
            # elif (v_type == "room" and u_type in ["agent", "asset", "object"]):
            #     sayplan_graph["links"].append(f"{v}↔{u}")

            # agent (user 1 ↔ anyone)
            if u == "user 1" or v == "user 1":
                sayplan_graph["links"].append(f"{u}↔{v}")
            # room-asset or asset-room
            elif (u_type == "room" and v_type == "asset") or \
               (u_type == "asset" and v_type == "room"):
                sayplan_graph["links"].append(f"{u}↔{v}")

            # asset-object or object-asset
            elif (u_type == "asset" and v_type == "object") or \
                 (u_type == "object" and v_type == "asset"):
                sayplan_graph["links"].append(f"{u}↔{v}")

        return sayplan_graph

    def _copy_edges_between(self, source_graph, target_graph, u, v):
        """Copy all edges between u↔v from source_graph to target_graph"""
        if source_graph.has_edge(u, v):
            for key, attr in source_graph.get_edge_data(u, v).items():
                target_graph.add_edge(u, v, **attr)
        if source_graph.has_edge(v, u):
            for key, attr in source_graph.get_edge_data(v, u).items():
                target_graph.add_edge(v, u, **attr)


    def task_relevant_graph_rebuild(self, sim_graph, task_data):

        raise NotImplementedError