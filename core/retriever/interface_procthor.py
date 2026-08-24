import os
import sys
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import re
from copy import deepcopy



    
class SceneGraphInterface_One():
    _DEFAULT_OUTPUT_KEYS = ['name_id_nl', 'node_type', 'properties', 'states']

    def _get_filtered_node_data(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        filtered_data = {}
        # name_id_nl (string 변환)
        if 'name_id_nl' in node_data and isinstance(node_data['name_id_nl'], (list, tuple)):
            filtered_data['name_id_nl'] = f"{node_data['name_id_nl'][0]} {node_data['name_id_nl'][1]}"
        else:
            filtered_data['name_id_nl'] = None
        # node_type 그대로
        filtered_data['node_type'] = node_data.get('node_type', None)
        # properties: True인 속성만 추출
        properties_keys = [
            'toggleable', 'canFillWithLiquid', 'cookable', 'breakable',
            'isHeatSource', 'isColdSource', 'sliceable', 
            'openable', 'pickupable'
        ]
        filtered_data['properties'] = [
            prop for prop in properties_keys if node_data.get(prop, False) == True
        ]
        # states: True인 상태만 추출
        states_keys = [
            'isToggled', 'isFilledWithLiquid', 'isCooked',
            'isSliced', 'isOpen', 'isPickedUp', 'isBroken'
        ]
        filtered_data['states'] = [
            state for state in states_keys if node_data.get(state, False) == True
        ]
        return filtered_data
    
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
                if k not in data:
                    return False
                data_val = data[k]
                if isinstance(data_val, list):
                    v_list = _as_list(v)
                    if not set(v_list).issubset(set(data_val)):
                        return False
                else:
                    if data_val != v:
                        return False
            return True
 
        G = scene_graph
        matches: List[str] = []
        for nid, ndata in G.nodes(data=True):
            if _match_label(nid, ndata) and _match_attrs(ndata):
                filtered_ndata = self._get_filtered_node_data(ndata)
                if 'room' in ndata:
                    filtered_ndata['room'] = ndata['room']
                matches.append(filtered_ndata)

        return matches
    
    def read_node(
        self,
        scene_graph: nx.MultiDiGraph,
        filt: Union[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Supports:
          - read_node({"node_id": "book 1"})
          - read_node({"node_id": "book 1", "attrs": [...], "default": ...})
        """
        properties_keys = [
            'toggleable', 'canFillWithLiquid', 'cookable', 'breakable',
            'isHeatSource', 'isColdSource', 'sliceable', 
            'openable', 'pickupable'
        ]
        states_keys = [
            'isToggled', 'isFilledWithLiquid', 'isCooked',
            'isSliced', 'isOpen', 'isPickedUp', 'isBroken'
        ]
        # Parse input
        if isinstance(filt, str):
            label = filt
            attrs = None
            default = None
        elif isinstance(filt, dict):
            label = filt.get("node_id")
            attrs = filt.get("attrs", None)
            default = filt.get("default", None)
            if label is None:
                return {}
        else:
            return {}

        # Parse label into (name, idx)
        try:
            name_part, idx_part = label.rsplit(" ", 1)
            name_part = name_part.strip().lower()
            idx_part = int(idx_part)
        except Exception:
            return {}  # invalid label format

        # Search for node whose name_id_nl matches (name, idx)
        G = scene_graph
        target_node_id = None
        for nid, data in G.nodes(data=True):
            if "name_id_nl" in data and isinstance(data["name_id_nl"], (list, tuple)):
                name_id = data["name_id_nl"]
                if name_id[0].lower() == name_part and int(name_id[1]) == idx_part:
                    target_node_id = nid
                    break

        if target_node_id is None:
            return {}

        attr_copy: Dict[str, Any] = deepcopy(G.nodes[target_node_id])

        if attrs is None:
            # Consistent formatting with _get_filtered_node_data
            filtered_data = {}
            if 'name_id_nl' in attr_copy and isinstance(attr_copy['name_id_nl'], (list, tuple)):
                filtered_data['name_id_nl'] = f"{attr_copy['name_id_nl'][0]} {attr_copy['name_id_nl'][1]}"
            else:
                filtered_data['name_id_nl'] = None
            filtered_data['salientMaterials'] = attr_copy.get("salientMaterials", [])
            filtered_data['properties'] = [
                k for k in properties_keys if attr_copy.get(k, False) is True
            ]
            filtered_data['states'] = [
                k for k in states_keys if attr_copy.get(k, False) is True
            ]
            return filtered_data

        if isinstance(attrs, str):
            attrs = [attrs]

        return {k: attr_copy.get(k, default) for k in attrs}

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
                    child_node_ids.add(u) # u가 asset 자식
            # Out-edges: room -> asset (edge_type 'r2a')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'r2a':
                    child_node_ids.add(v) # v가 asset 자식

        elif parent_node_type == 'asset':
            # In-edges: object -> asset (edge_type 'o2a')
            for u, _, edge_data in scene_graph.in_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'o2a':
                    child_node_ids.add(u) # u가 object 자식
            # Out-edges: asset -> object (edge_type 'a2o')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'a2o':
                    child_node_ids.add(v) # v가 object 자식
        
        children_data_list: List[Dict[str, Any]] = []
        for child_id in child_node_ids:
            if child_id in scene_graph: # 자식 노드가 그래프에 실제로 존재하는지 확인
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
                    child_node_ids.add(u) # u가 asset 자식
            # Out-edges: room -> asset (edge_type 'r2a')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'r2a':
                    child_node_ids.add(v) # v가 asset 자식

        elif parent_node_type == 'asset':
            # In-edges: object -> asset (edge_type 'o2a')
            for u, _, edge_data in scene_graph.in_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'o2a':
                    child_node_ids.add(u) # u가 object 자식
            # Out-edges: asset -> object (edge_type 'a2o')
            for _, v, edge_data in scene_graph.out_edges(parent_node_id, data=True):
                if edge_data.get('edge_type') == 'a2o':
                    child_node_ids.add(v) # v가 object 자식
        
        def _as_list(x: Union[str, List[str]]) -> List[str]:
            return x if isinstance(x, list) else [x]
        
        def _match_attrs(data: Dict[str, Any]) -> bool:
            for k, v in child_node_attrs.items():
                if k not in data:
                    return False
                data_val = data[k]
                if isinstance(data_val, list):
                    v_list = _as_list(v)
                    if not set(v_list).issubset(set(data_val)):
                        return False
                else:
                    if data_val != v:
                        return False
            return True

        children_data_list: List[Dict[str, Any]] = []
        for child_id in child_node_ids:
            if child_id in scene_graph: # 자식 노드가 그래프에 실제로 존재하는지 확인
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
    
    def get_edges_for_node(
            self, 
            scene_graph: nx.DiGraph, 
            node_id: str, 
            relation: Optional[str] = None
        ) -> List[Tuple[str, str, str]]:
        edge_info_list = []
        # Outgoing edges: node_id → other
        for _, target, edge_data in scene_graph.out_edges(node_id, data=True):
            rel = edge_data.get("relation")
            if relation is None or rel == relation:
                edge_info_list.append((node_id, target, rel))
        # Incoming edges: other → node_id
        for source, _, edge_data in scene_graph.in_edges(node_id, data=True):
            rel = edge_data.get("relation")
            if relation is None or rel == relation:
                edge_info_list.append((source, node_id, rel))
        return edge_info_list
    
def compress_object_list(object_list: str) -> str:
    # 1. object_list를 comma로 나누고 strip
    # entries = [entry.strip() for entry in object_list.split(",") if entry.strip()]
    entries = object_list
    # 2. 딕셔너리: 클래스 이름 → 번호 리스트
    grouped = defaultdict(list)
    for entry in entries:
        match = re.match(r"(.+?)\s+(\d+)$", entry)
        if match:
            cls, idx = match.group(1), int(match.group(2))
            grouped[cls].append(idx)
        else:
            # 매치되지 않는 경우 그냥 문자열 그대로 저장
            grouped[entry].append(None)

    # 3. 클래스별로 번호가 여러 개면 압축
    compressed_entries = []
    for cls, ids in grouped.items():
        ids = sorted(i for i in ids if i is not None)
        if len(ids) > 1:
            compressed_entries.append(f"{cls} ({', '.join(str(i) for i in ids)})")
        elif len(ids) == 1:
            compressed_entries.append(f"{cls} {ids[0]}")
        else:
            compressed_entries.append(cls)  # None만 있는 경우s

    return ", ".join(compressed_entries)