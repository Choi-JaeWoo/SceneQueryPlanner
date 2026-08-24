import os
import sys
import networkx as nx


from core.wm.sg_base import BaseSG
import matplotlib.pyplot as plt

class ProcThorSG(BaseSG):
    """
    Abstract base class for a 3D Scene Graph.
    """

    def __init__(self, cfg, init_graph, name_id_dict_sim2nl, name_id_dict_nl2sim):
        """
        Initialize the planner with a simulation environment and an LLM-based decision-making agent.
        """
        self.cfg = cfg
        self.name_id_dict_sim2nl = name_id_dict_sim2nl
        self.name_id_dict_nl2sim = name_id_dict_nl2sim
        self.scene_graph = self.convert_nx_graph(init_graph)

    def convert_nx_graph(self, sim_graph):
        G = nx.MultiDiGraph()
        name_id_dict_sim2nl = self.name_id_dict_sim2nl
        id2node = {node['id']: node for node in sim_graph.get("nodes", [])}
        for node in sim_graph.get("nodes", []):
            name_id_sim = (node["objectType"], node["objectId"])
            if name_id_sim not in name_id_dict_sim2nl:
                continue
            name_id_nl = name_id_dict_sim2nl[name_id_sim]
            node_key = f"{name_id_nl[0]} {name_id_nl[1]}"

            category = node.get("category", "")
            if category == "Rooms":
                node_type = "room"
            elif category == "asset":
                node_type = "asset"
            elif category == "object":
                node_type = "object"
            else:
                node_type = "unknown"

            sg_node = {
                'name_id_sim': name_id_sim,
                'name_id_nl': name_id_nl,
                'node_type': node_type
            }
            sg_node.update(node)
            G.add_node(node_key, **sg_node)
        
        for edge in sim_graph.get("edges", []):
            head_name_id_sim = (id2node[edge["from_id"]]["objectType"], id2node[edge["from_id"]]["objectId"])
            tail_name_id_sim = (id2node[edge["to_id"]]["objectType"], id2node[edge["to_id"]]["objectId"])
            
            if head_name_id_sim not in name_id_dict_sim2nl or tail_name_id_sim not in name_id_dict_sim2nl:
                continue

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
            
        return G
    
    def update_nx_graph(self, partial_graph, last_event):
        G_new = self.convert_nx_graph(partial_graph)
        G = self.scene_graph  # existing full scene graph

        # --- Node update ---
        for node_key, attrs in G_new.nodes(data=True):
            if node_key not in G:
                G.add_node(node_key, **attrs)
            else:
                for k, v in attrs.items():
                    if G.nodes[node_key].get(k) != v:
                        G.nodes[node_key][k] = v  # update state/attributes

        # --- Edge update ---
        # 1. Remove existing edges of the user node
        user_key = "user 1"
        if user_key in G:
            for u, v, k in list(G.out_edges(user_key, keys=True)):
                G.remove_edge(u, v, k)
            for u, v, k in list(G.in_edges(user_key, keys=True)):
                G.remove_edge(u, v, k)
        # 2. Remove existing edges of held_nodes
        inventory = last_event.metadata.get('inventoryObjects', [])
        held_node_key = None
        connected_objs = set()
        if inventory:
            held_node = inventory[0]
            held_name_id_sim = (held_node["objectType"].lower(), held_node['objectId'])
            held_name_id_nl = self.name_id_dict_sim2nl[held_name_id_sim]
            if held_name_id_nl:
                held_node_key = f"{held_name_id_nl[0]} {held_name_id_nl[1]}"
                if held_node_key in G:
                    # remove out-edges
                    for u, v, k in list(G.out_edges(held_node_key, keys=True)):
                        G.remove_edge(u, v, k)
                    # remove in-edges
                    for u, v, k in list(G.in_edges(held_node_key, keys=True)):
                        edge_data = G[u][v][k]
                        if edge_data.get("relation") in {"ON", "INSIDE"}:
                            continue  # keep objects inside the receptacle
                        G.remove_edge(u, v, k)
                ################  
                for u, v, k, data in G.in_edges(held_node_key, keys=True, data=True):
                    if data.get("relation") in {"ON", "INSIDE"}:
                        connected_objs.add(u)
                        
                for obj_key in connected_objs:
                    for uu, vv, kk in list(G.in_edges(obj_key, keys=True)):
                        if vv == held_node_key:
                            continue  # keep relations with held_node
                        G.remove_edge(uu, vv, kk)
                    for uu, vv, kk in list(G.out_edges(obj_key, keys=True)):
                        if vv == held_node_key:
                             continue  # keep relations with held_node
                        G.remove_edge(uu, vv, kk)
                        
                if user_key in G:
                    G.add_edge(user_key, held_node_key, relation="HOLD")
            
        # 3. General node update: remove existing edges if both u and v are in partial ---
        affected_pairs = {
            (u, v) for u, v, k in G_new.edges(keys=True) if u in G and v in G and u != held_node_key and v != held_node_key
        }
        for u, v in affected_pairs:
            if G.has_edge(u, v):
                for k in list(G[u][v].keys()):
                    G.remove_edge(u, v, k)

        # 4. Add edges from partial_graph ---
        for u, v, k, attrs in G_new.edges(keys=True, data=True):
            if u in G and v in G and not u in connected_objs and v not in connected_objs:
                G.add_edge(u, v, **attrs)

        self.scene_graph = G
        
        
class ProcThorSG_One(BaseSG):
    def __init__(self, cfg, init_graph, name_id_dict_sim2nl, name_id_dict_nl2sim):
        """
        Initialize the planner with a simulation environment and an LLM-based decision-making agent.
        """
        self.cfg = cfg
        self.name_id_dict_sim2nl = name_id_dict_sim2nl
        self.name_id_dict_nl2sim = name_id_dict_nl2sim
        self.scene_graph = self.convert_nx_graph(init_graph, initialize=True)
        ### visited initialization
        for node_key in self.scene_graph.nodes:
            self.scene_graph.nodes[node_key]['visited'] = False
        for source, target, edge_data in self.scene_graph.out_edges('user 1', data=True):
            if self.scene_graph.nodes[target]['node_type'] == 'room':
                self.scene_graph.nodes[target]['visited'] = True
        
        # (1) Collect room and asset nodes
        room_and_asset_nodes = {node_key for node_key, data in self.scene_graph.nodes(data=True)
                                if data.get('node_type') in ['room', 'asset'] or node_key == 'user 1'}

        # (2) Collect nodes connected to 'user 1' (regardless of edge direction)
        connected_to_agent = set()
        for u, v in self.scene_graph.in_edges('user 1'):
            connected_to_agent.add(u)
        for u, v in self.scene_graph.out_edges('user 1'):
            connected_to_agent.add(v)
            
        # (2.5) Add object nodes not connected to any asset via INSIDE/ON edges
        unconnected_objects = set()
        for node_key, data in self.scene_graph.nodes(data=True):
            if data.get("node_type") != "object":
                continue
            connected = False  # whether connected to an asset via INSIDE/ON
            for a, adata in self.scene_graph.nodes(data=True):
                if adata.get("node_type") != "asset":
                    continue
                # direction: object → asset
                if self.scene_graph.has_edge(node_key, a):
                    edge_dict = self.scene_graph.get_edge_data(node_key, a)
                    for edge_data in edge_dict.values():
                        if edge_data.get("relation") in ["INSIDE", "ON"]:
                            connected = True
                            break
                    if connected:
                        break
                # direction: asset → object
                if self.scene_graph.has_edge(a, node_key):
                    edge_dict = self.scene_graph.get_edge_data(a, node_key)
                    for edge_data in edge_dict.values():
                        if edge_data.get("relation") in ["INSIDE", "ON"]:
                            connected = True
                            break
                    if connected:
                        break
            if not connected:
                unconnected_objects.add(node_key)

        # (3) Final node set
        faucet_nodes = {
            node_key for node_key, data in self.scene_graph.nodes(data=True)
            if "faucet" in node_key.lower()
        }
        final_nodes = room_and_asset_nodes.union(connected_to_agent).union(unconnected_objects).union(faucet_nodes)

        # (4) Create induced subgraph and replace
        self.scene_graph = self.scene_graph.subgraph(final_nodes).copy()



    def convert_nx_graph(self, sim_graph, initialize=False):
        G = nx.MultiDiGraph()
        name_id_dict_sim2nl = self.name_id_dict_sim2nl
        id2node = {node['id']: node for node in sim_graph.get("nodes", [])}
        
        for node in sim_graph.get("nodes", []):
            name_id_sim = (node["objectType"], node["objectId"])
            if name_id_sim not in name_id_dict_sim2nl:
                continue
            name_id_nl = name_id_dict_sim2nl[name_id_sim]
            node_key = f"{name_id_nl[0]} {name_id_nl[1]}"
            category = node.get("category", "")
            if category == "Rooms":
                node_type = "room"
            elif category == "asset":
                node_type = "asset"
            elif category == "object":
                node_type = "object"
            else:
                node_type = "unknown"

            sg_node = {
                'name_id_sim': name_id_sim,
                'name_id_nl': name_id_nl,
                'node_type': node_type
            }
            sg_node.update(node)
            G.add_node(node_key, **sg_node)
        
        for edge in sim_graph.get("edges", []):
            head_name_id_sim = (id2node[edge["from_id"]]["objectType"], id2node[edge["from_id"]]["objectId"])
            tail_name_id_sim = (id2node[edge["to_id"]]["objectType"], id2node[edge["to_id"]]["objectId"])
            
            if head_name_id_sim not in name_id_dict_sim2nl or tail_name_id_sim not in name_id_dict_sim2nl:
                continue

            head_name_id_nl = name_id_dict_sim2nl[head_name_id_sim]
            tail_name_id_nl = name_id_dict_sim2nl[tail_name_id_sim]
            
            head_category = id2node[edge["from_id"]].get("category", "")
            tail_category = id2node[edge["to_id"]].get("category", "")
            if head_category == "Rooms":
                head_node_type = "room"
            elif head_category == "asset":
                head_node_type = "asset"
            elif head_category == "object":
                head_node_type = "object"
            else:
                head_node_type = "unknown"

            if tail_category == "Rooms":
                tail_node_type = "room"
            elif tail_category == "asset":
                tail_node_type = "asset"
            elif tail_category == "object":
                tail_node_type = "object"
            else:
                tail_node_type = "unknown"
            
            if (head_node_type, tail_node_type) == ("room", "room"):
                edge_type = 'r2r'
            elif (head_node_type, tail_node_type) == ("room", "asset"):
                edge_type = 'r2a'
            elif (head_node_type, tail_node_type) == ("room", "object"):
                edge_type = 'r2o'
            elif (head_node_type, tail_node_type) == ("asset", "room"):
                edge_type = 'a2r'
            elif (head_node_type, tail_node_type) == ("asset", "asset"):
                edge_type = 'a2a'
            elif (head_node_type, tail_node_type) == ("asset", "object"):
                edge_type = 'a2o'
            elif (head_node_type, tail_node_type) == ("object", "room"):
                edge_type = 'o2r'
            elif (head_node_type, tail_node_type) == ("object", "asset"):
                edge_type = 'o2a'
            elif (head_node_type, tail_node_type) == ("object", "object"):
                edge_type = 'o2o'
            else:
                edge_type = 'unknown'
            

            sg_edge = {
                'relation': edge.get("relation_type", ""),
                'edge_type': edge_type,
                'head_name_id_nl': head_name_id_nl,
                'tail_name_id_nl': tail_name_id_nl,
                'head_name_id_sim': head_name_id_sim,
                'tail_name_id_sim': tail_name_id_sim,
            }

            head_key = f"{head_name_id_nl[0]} {head_name_id_nl[1]}"
            tail_key = f"{tail_name_id_nl[0]} {tail_name_id_nl[1]}"

            G.add_edge(head_key, tail_key, **sg_edge)
        if initialize == True:
            for room_key in G.nodes:
                if G.nodes[room_key].get("node_type") == "room":
                    # for all incoming edges connected to the room node
                    for source_key, _, edge_data in G.in_edges(room_key, data=True):
                        if G.nodes[source_key].get("node_type") == "asset":
                            G.nodes[source_key]["room"] = room_key

        return G
    
    def update_nx_graph(self, partial_graph, last_event, visited_node=None):    
        G_new = self.convert_nx_graph(partial_graph)
        #########################################################################
        # (1) Collect room and asset nodes
        room_asset_agent_nodes = {node_key for node_key, data in G_new.nodes(data=True)
                                if data.get('node_type') in ['room', 'asset'] or node_key == 'user 1'}

        # (2) Collect nodes connected to 'user 1' via InFront
        connected_to_agent = set()
        infront_assets = set()
        for _, v, edge_data in G_new.out_edges('user 1', data=True):
            if edge_data.get("relation") == "InFront":
                connected_to_agent.add(v)
                infront_assets.add(v)

        # (2.1) Also collect objects inside assets connected via InFront
        for asset_node in infront_assets:
            for obj_node, _, e_data in G_new.in_edges(asset_node, data=True):
                if e_data.get("edge_type") == "o2a":
                    connected_to_agent.add(obj_node)
                    
        # (2.2) Collect nodes connected to 'user 1' (regardless of edge direction)
        for u, v in G_new.in_edges('user 1'):
            connected_to_agent.add(u)
        for u, v in G_new.out_edges('user 1'):
            connected_to_agent.add(v)

        # (2.2) Delete InFront edges
        edges_to_remove = []
        for u, v, edge_data in G_new.edges(data=True):
            if edge_data.get("relation") == "InFront":
                edges_to_remove.append((u, v))
        G_new.remove_edges_from(edges_to_remove)
        # all included nodes
        final_nodes = room_asset_agent_nodes.union(connected_to_agent)

        # --- Refresh G_new as the subgraph ---
        G_new = G_new.subgraph(final_nodes).copy()
        ###########################################################################

        G = self.scene_graph  # existing full scene graph
        
        # --- Node update ---
        for node_key, attrs in G_new.nodes(data=True):
            if node_key not in G:
                G.add_node(node_key, **attrs)
                for node_key in G.nodes:
                    if 'visited' not in G.nodes[node_key]:
                        G.nodes[node_key]['visited'] = False
            else:
                for k, v in attrs.items():
                    if G.nodes[node_key].get(k) != v:
                        G.nodes[node_key][k] = v  # update state/attributes
        
        # --- Edge update ---
        # 1. Remove existing edges of the user node
        user_key = "user 1"
        if user_key in G:
            for u, v, k in list(G.out_edges(user_key, keys=True)):
                G.remove_edge(u, v, k)
            for u, v, k in list(G.in_edges(user_key, keys=True)):
                G.remove_edge(u, v, k)
        # 2. Remove existing edges of held_nodes
        inventory = last_event.metadata.get('inventoryObjects', [])
        held_node_key = None
        connected_objs = set()
        if inventory:
            held_node = inventory[0]
            held_name_id_sim = (held_node["objectType"].lower(), held_node['objectId'])
            held_name_id_nl = self.name_id_dict_sim2nl[held_name_id_sim]
            if held_name_id_nl:
                held_node_key = f"{held_name_id_nl[0]} {held_name_id_nl[1]}"
                if held_node_key in G:
                    # remove out-edges
                    for u, v, k in list(G.out_edges(held_node_key, keys=True)):
                        G.remove_edge(u, v, k)
                    # remove in-edges
                    for u, v, k in list(G.in_edges(held_node_key, keys=True)):
                        edge_data = G[u][v][k]
                        if edge_data.get("relation") in {"ON", "INSIDE"}:
                            continue  # keep objects inside the receptacle
                        G.remove_edge(u, v, k)
                ################  
                for u, v, k, data in G.in_edges(held_node_key, keys=True, data=True):
                    if data.get("relation") in {"ON", "INSIDE"}:
                        connected_objs.add(u)
                        
                for obj_key in connected_objs:
                    for uu, vv, kk in list(G.in_edges(obj_key, keys=True)):
                        if vv == held_node_key:
                            continue  # keep relations with held_node
                        G.remove_edge(uu, vv, kk)
                    for uu, vv, kk in list(G.out_edges(obj_key, keys=True)):
                        if vv == held_node_key:
                             continue  # keep relations with held_node
                        G.remove_edge(uu, vv, kk)
                        
                if user_key in G:
                    G.add_edge(user_key, held_node_key, relation="HOLD")
            
        # 3. General node update: remove existing edges if both u and v are in partial ---
        affected_pairs = {
            (u, v) for u, v, k in G_new.edges(keys=True) if u in G and v in G and u != held_node_key and v != held_node_key
        }
        for u, v in affected_pairs:
            if G.has_edge(u, v):
                for k in list(G[u][v].keys()):
                    G.remove_edge(u, v, k)

        # 4. Add edges from partial_graph ---
        for u, v, k, attrs in G_new.edges(keys=True, data=True):
            if u in G and v in G and not u in connected_objs and v not in connected_objs:
                G.add_edge(u, v, **attrs)
        
        if visited_node is not None and visited_node in G.nodes:
            G.nodes[visited_node]['visited'] = True
        
        self.scene_graph = G


def draw_scene_graph(G, save_path, title="Packed Subgraph Layout", user_node="user 1"):
    # 1. connected components (for directed graph: weakly)
    components = list(nx.weakly_connected_components(G))

    # 2. Initialize graph layout
    all_pos = dict()
    x_offset = 0

    plt.figure(figsize=(16, 12))
    plt.title(title)

    for i, comp in enumerate(components):
        subG = G.subgraph(comp)
        # each component gets an independent layout
        pos = nx.spring_layout(subG, seed=42, k=0.5)  # keep internal nodes clustered
        # shift position along x-axis (packing)
        pos = {n: (x + x_offset, y) for n, (x, y) in pos.items()}
        x_offset += 3.0  # shift next component to the right

        all_pos.update(pos)

        # set node colors
        node_colors = ["orange" if n == user_node else "skyblue" for n in subG.nodes]
        nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=800)
        nx.draw_networkx_labels(subG, pos, font_size=8)

        # edges and labels
        for u, v, k, data in subG.edges(keys=True, data=True):
            rel = data.get("relation", "rel")
            rad = 0.15 + 0.05 * (k % 4)
            nx.draw_networkx_edges(
                subG, pos, edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad}",
                arrows=True,
                arrowstyle="-|>"
            )
            nx.draw_networkx_edge_labels(
                subG, pos,
                edge_labels={(u, v): f"{rel} ({k})"},
                font_size=6,
                label_pos=0.6,
                font_color="red"
            )

    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    
    plt.savefig(save_path, dpi=300)
    plt.close()