import os
import sys
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import re
from copy import deepcopy



class SceneGraphInterface():    
    # ---------------------------------------------------------------------
    # 1. Object lookup ------------------------------------------------------
    # ---------------------------------------------------------------------
    def find_objects(self, scene_graph, filt: Dict[str, Any]) -> List[str]:
        """Return *node‑ids* that satisfy ``filt``.

        **Filter schema** (all keys optional)::

            {
                "label": str | List[str],       # class name(s) – case‑insensitive
                "attrs": {
                    "properties": str|List[str],# node["properties"] contains *all*
                    "states": str|List[str],    # node["states"] contains *all*
                    <any other key>: value,      # exact match on node[key]
                }
            }

        The function is intentionally tolerant:
        * Unknown filter keys are ignored.
        * Single values or lists are both accepted for label / attrs.
        """
        label_filter: Union[str, List[str], None] = filt.get("label")
        attrs_filter: Dict[str, Any] = filt.get("attrs", {})

        if isinstance(label_filter, str):
            label_filter = [label_filter]
        if label_filter is not None:
            label_filter = [lbl.lower() for lbl in label_filter]
        
        def _match_label(node_id: str, data: Dict[str, Any]) -> bool:
            if label_filter is None:
                return True
            # Priority 1 – explicit natural‑language name if present
            if "name_id_nl" in data and isinstance(data["name_id_nl"], (list, tuple)):
                node_lbl = str(data["name_id_nl"][0]).lower()
            else:
                node_lbl = node_id.split()[0].lower()
            return node_lbl in label_filter

        def _as_list(x: Union[str, List[str]]) -> List[str]:
            return x if isinstance(x, list) else [x]

        def _match_attrs(data: Dict[str, Any]) -> bool:
            for k, v in attrs_filter.items():
                if k in {"properties", "property"}:
                    props = set(data.get("properties", []))
                    if not set(_as_list(v)).issubset(props):
                        return False
                elif k in {"states", "state"}:
                    states = set(data.get("states", []))
                    if not set(_as_list(v)).issubset(states):
                        return False
                else:  # exact value match
                    if data.get(k) != v:
                        return False
            return True
        
        G = scene_graph
        matches: List[str] = []
        for nid, ndata in G.nodes(data=True):
            if _match_label(nid, ndata) and _match_attrs(ndata):
                matches.append(ndata)
        
        return matches
    
    # ------------------------------------------------------------------
    # 2. Node read -------------------------------------------------------
    # ------------------------------------------------------------------
    def read_node(
        self,
        scene_graph: nx.MultiDiGraph,
        node_id: str,
        keys: Optional[Union[str, List[str]]] = None,
        default: Any = None,
    ) -> Dict[str, Any]:
        """Return *a shallow copy* of the requested node's attribute dict.

        Parameters
        ----------
        scene_graph : nx.MultiDiGraph
            Graph to query. Latest snapshot should be supplied by caller.
        node_id : str
            Exact node identifier (e.g. ``"plate 7"``).
        keys : str | list[str] | None, optional
            If provided, filter the returned dict to these attribute names
            (case‑sensitive exact match). Single string will be coerced to a
            one‑element list. ``None`` ⇒ return **all** attributes.
        default : Any, optional
            Value to insert for missing attributes when *keys* is specified.
            Ignored when *keys* is ``None``.

        Returns
        -------
        Dict[str, Any]
            JSON‑serialisable copy of the node's attributes.
            Empty dict if *node_id* does not exist.
        """
        G = scene_graph
        if node_id not in G:
            return {}

        attr_copy: Dict[str, Any] = deepcopy(G.nodes[node_id])

        if keys is None:
            return attr_copy

        # keys filtering
        if isinstance(keys, str):
            keys = [keys]

        return {k: attr_copy.get(k, default) for k in keys}


class SceneGraphInterface_One():
    _DEFAULT_OUTPUT_KEYS = ['name_id_nl', 'node_type', 'properties', 'states']

    def _get_filtered_node_data(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: node_data.get(key) for key in self._DEFAULT_OUTPUT_KEYS
        }
    
    def find_objects(self, scene_graph, filt: Dict[str, Any]) -> List[str]:
        """Return *node‑ids* that satisfy ``filt``.

        **Filter schema** (all keys optional)::

            {
                "label": str | List[str],       # class name(s) – case‑insensitive
                "attrs": {
                    "properties": str|List[str],# node["properties"] contains *all*
                    "states": str|List[str],    # node["states"] contains *all*
                    <any other key>: value,      # exact match on node[key]
                }
            }

        The function is intentionally tolerant:
        * Unknown filter keys are ignored.
        * Single values or lists are both accepted for label / attrs.
        """
        label_filter: Union[str, List[str], None] = filt.get("label")
        attrs_filter: Dict[str, Any] = filt.get("attrs", {})

        if isinstance(label_filter, str):
            label_filter = [label_filter]
        if label_filter is not None:
            label_filter = [lbl.lower() for lbl in label_filter]
        
        def _match_label(node_id: str, data: Dict[str, Any]) -> bool:
            if label_filter is None:
                return True
            # Priority 1 – explicit natural‑language name if present
            if "name_id_nl" in data and isinstance(data["name_id_nl"], (list, tuple)):
                node_lbl = str(data["name_id_nl"][0]).lower()
            else:
                node_lbl = node_id.split()[0].lower()
            return node_lbl in label_filter

        def _as_list(x: Union[str, List[str]]) -> List[str]:
            return x if isinstance(x, list) else [x]

        def _match_attrs(data: Dict[str, Any]) -> bool:
            for k, v in attrs_filter.items():
                if k in {"properties", "property"}:
                    props = set(data.get("properties", []))
                    if not set(_as_list(v)).issubset(props):
                        return False
                elif k in {"states", "state"}:
                    states = set(data.get("states", []))
                    if not set(_as_list(v)).issubset(states):
                        return False
                else:  # exact value match
                    if data.get(k) != v:
                        return False
            return True
        
        G = scene_graph
        matches: List[str] = []
        for nid, ndata in G.nodes(data=True):
            if _match_label(nid, ndata) and _match_attrs(ndata):
                filtered_ndata = {key: ndata.get(key) for key in ['name_id_nl', 'node_type', 'properties', 'states']}
                if 'room' in ndata:
                    filtered_ndata['room'] = ndata['room']
                filtered_ndata['name_id_nl'] = f"{filtered_ndata['name_id_nl'][0]} {filtered_ndata['name_id_nl'][1]}"
                matches.append(filtered_ndata)    
        return matches

    def get_child_nodes(self, 
                                scene_graph: nx.DiGraph, 
                                parent_node_id: str, 
                                target_node_type: Optional[str] = None,
                                max_depth: Optional[int] = None) -> List[Dict[str, Any]]:
        parent_node_data = scene_graph.nodes.get(parent_node_id)
        parent_node_type = parent_node_data.get("node_type")
        child_node_ids = set()

        if parent_node_type == 'room':
            # In-edges: asset -> room (edge_type 'a2r')
            for u, _, edge_data in scene_graph.in_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'a2r':
                    child_node_ids.add(u) # u is the asset child
            # Out-edges: room -> asset (edge_type 'r2a')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'r2a':
                    child_node_ids.add(v) # v is the asset child

        elif parent_node_type == 'asset':
            # In-edges: object -> asset (edge_type 'o2a')
            for u, _, edge_data in scene_graph.in_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'o2a':
                    child_node_ids.add(u) # u is the object child
            # Out-edges: asset -> object (edge_type 'a2o')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'a2o':
                    child_node_ids.add(v) # v is the object child
        
        children_data_list: List[Dict[str, Any]] = []
        for child_id in child_node_ids:
            if child_id in scene_graph: # check that the child node actually exists in the graph
                child_ndata = scene_graph.nodes[child_id]
                filtered_data = self._get_filtered_node_data(child_ndata)
                children_data_list.append(filtered_data)
            
        return children_data_list

    def get_child_node_names(self, 
                                scene_graph: nx.DiGraph, 
                                parent_node_id: str, 
                                child_node_attrs = None,
                                target_node_type: Optional[str] = None,
                                max_depth: Optional[int] = None) -> List[Dict[str, Any]]:
        parent_node_data = scene_graph.nodes.get(parent_node_id)
        parent_node_type = parent_node_data.get("node_type")
        child_node_ids = set()

        if parent_node_type == 'room':
            # In-edges: asset -> room (edge_type 'a2r')
            for u, _, edge_data in scene_graph.in_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'a2r':
                    child_node_ids.add(u) # u is the asset child
            # Out-edges: room -> asset (edge_type 'r2a')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'r2a':
                    child_node_ids.add(v) # v is the asset child

        elif parent_node_type == 'asset':
            # In-edges: object -> asset (edge_type 'o2a')
            for u, _, edge_data in scene_graph.in_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'o2a':
                    child_node_ids.add(u) # u is the object child
            # Out-edges: asset -> object (edge_type 'a2o')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'a2o':
                    child_node_ids.add(v) # v is the object child
        
        def _as_list(x: Union[str, List[str]]) -> List[str]:
            return x if isinstance(x, list) else [x]
        
        def _match_attrs(data: Dict[str, Any]) -> bool:
            for k, v in child_node_attrs.items():
                if k in {"properties", "property"}:
                    props = set(data.get("properties", []))
                    if not set(_as_list(v)).issubset(props):
                        return False
                elif k in {"states", "state"}:
                    states = set(data.get("states", []))
                    if not set(_as_list(v)).issubset(states):
                        return False
                else:  # exact value match
                    if data.get(k) != v:
                        return False
            return True

        children_data_list: List[Dict[str, Any]] = []
        for child_id in child_node_ids:
            if child_id in scene_graph: # check that the child node actually exists in the graph
                child_ndata = scene_graph.nodes[child_id]
                child_node_name =  f"{child_ndata['name_id_nl'][0]} {child_ndata['name_id_nl'][1]}"
                
                if child_node_attrs is not None:
                    if _match_attrs(child_ndata):
                        children_data_list.append(child_node_name) 
                    
                else:
                    children_data_list.append(child_node_name)
                children_data_list.sort()
        
        x = compress_object_list(children_data_list)
        
        return x
    
    def get_edges_for_node(self, scene_graph: nx.DiGraph, node_id: str) -> List[Tuple[str, str, str]]:
        edge_info_list = []
        # Outgoing edges: node_id → other
        for _, target, edge_data in scene_graph.out_edges(node_id, data=True):
            edge_info_list.append((node_id, target, edge_data['relation']))

        # Incoming edges: other → node_id
        for source, _, edge_data in scene_graph.in_edges(node_id, data=True):
            edge_info_list.append((source, node_id, edge_data['relation']))
        return edge_info_list
    
def compress_object_list(object_list: str) -> str:
    # 1. Split object_list by commas and strip
    # entries = [entry.strip() for entry in object_list.split(",") if entry.strip()]
    entries = object_list
    # 2. Dictionary: class name -> list of numbers
    grouped = defaultdict(list)
    for entry in entries:
        match = re.match(r"(.+?)\s+(\d+)$", entry)
        if match:
            cls, idx = match.group(1), int(match.group(2))
            grouped[cls].append(idx)
        else:
            # If no match, store the string as-is
            grouped[entry].append(None)

    # 3. Compress when a class has multiple numbers
    compressed_entries = []
    for cls, ids in grouped.items():
        ids = sorted(i for i in ids if i is not None)
        if len(ids) > 1:
            compressed_entries.append(f"{cls} ({', '.join(str(i) for i in ids)})")
        elif len(ids) == 1:
            compressed_entries.append(f"{cls} {ids[0]}")
        else:
            compressed_entries.append(cls)  # case with only None values

    return ", ".join(compressed_entries)