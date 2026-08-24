import os
import sys
import ast # For safely evaluating string literals
import re # For parsing the interface call string
from datetime import datetime
from typing import Dict, Any, Optional # Added Optional
import networkx as nx

import inspect

from core.retriever.base_retriever import BaseRetriever
from core.retriever.interface_procthor import SceneGraphInterface_One


class ProcThorRetriever(BaseRetriever):
    def __init__(self, cfg, llm_agent, sg_interface):
        super().__init__(cfg, llm_agent, sg_interface)

    def run(self, question):
        raise NotImplementedError

    def collect_human(self, query, working_memory, collect_path):
        os.makedirs(os.path.dirname(collect_path), exist_ok=True)
        current_graph = working_memory.scene_graph
        with open(collect_path, 'w') as f:
            f.write(f'Query: {query}\n')
        print(f"\n\nQuery: {query}")
        while True:
            decision_type = input("Enter decision type (i for interface calling, r for reasoning, a for answering): ").strip().lower()
            if decision_type == 'i':
                interface_call_str = input("Enter interface call (e.g., \"find_objects({'label': 'plate', 'attrs': {'states': 'on_table'}})\"):\n")
                parsed_call = self._parse_interface_call_string(interface_call_str)
                
                with open(collect_path, 'a', encoding='utf-8') as f:
                    f.write(f'Interface Calling: {interface_call_str}\n')
                
                execution_outcome = self._execute_interface_call_from_parsed(parsed_call, current_graph)
                
                if execution_outcome['execution_successful']:
                    with open(collect_path, 'a', encoding='utf-8') as f:
                        f.write(f'Information: {execution_outcome["result"]}\n')
                    print(f'Information: {execution_outcome["result"]}')
                else:
                    with open(collect_path, 'a', encoding='utf-8') as f:
                        f.write(f'Error: {execution_outcome["error"]}\n')
                    print(f'Error: {execution_outcome["error"]}')
            elif decision_type == 'a':
                answer = input("Answer: ")
                with open(collect_path, 'a', encoding='utf-8') as f:
                    f.write(f'Answer: {answer}')
                print('\n\n')
                return answer
            elif decision_type == 'r':
                reasoning = input("Reasoning: ")
                with open(collect_path, 'a', encoding='utf-8') as f:
                    f.write(f'Think: {reasoning}\nOK.\n')
                # print(f'Think: {reasoning}')

    def collect_llm(self, question):
        raise NotImplementedError


    def _parse_interface_call_string(self, call_str: str) -> Optional[Dict[str, Any]]:
        """
        Parses a string like "function_name({'arg1': 'val1'})"
        Returns a dict like {"name": "function_name", "args_dict": {"arg1": "val1"}} or None on failure.
        """
        # Regex to capture function name and the arguments part (inside parentheses)
        # It expects arguments to be a single dictionary literal.
        match = re.fullmatch(r"\s*(\w+)\s*\((.*)\)\s*", call_str)
        if not match:
            print(f"Error: Could not parse function call structure from '{call_str}'. Expected format: func_name(args_dict_str)")
            return None

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # (a) When arguments are empty -> None
        if args_str == "":
            args_literal = None

        # (b) When arguments exist -> parse safely
        else:
            try:
                args_literal = ast.literal_eval(args_str)
            except (ValueError, SyntaxError):
                print(f"Error parsing arguments for '{func_name}': {args_str}")
                return None

        return {"name": func_name, "args": args_literal}
        
    def _execute_interface_call_from_parsed(self, parsed_call, current_graph):
        if parsed_call is None:
            outcome = {"execution_successful": False, "result": None, "error": None}
            outcome["error"] = "invalid query"
            return outcome
        func_name  = parsed_call["name"]
        args       = parsed_call["args"]      # str | dict | None
        outcome    = {"execution_successful": False, "result": None, "error": None}

        # (1) Validate the graph
        if current_graph is None:
            outcome["error"] = "working_memory.scene_graph is None"
            return outcome

        # (2) Check that the interface exists
        if not hasattr(self.sg_interface, func_name):
            outcome["error"] = f"SceneGraphInterface has no method '{func_name}'"
            return outcome

        method = getattr(self.sg_interface, func_name)

        try:
            # (3) Branch between find_objects and read_node
            if func_name == "find_objects":
                if not isinstance(args, dict):
                    raise ValueError("find_objects expects a dict argument.")
                result = method(current_graph, filt=args)

            elif func_name == "get_child_nodes": # explicit handling via get_descendant_nodes_data
                if not isinstance(args, dict):
                    raise ValueError(f"'{func_name}' expects a dict argument (e.g., {{'parent_node_id': 'id', ...}}).")
                # assumes signature get_descendant_nodes_data(self, scene_graph, parent_node_id, ...)
                # pass all key-value pairs of the 'args' dict as keyword arguments
                result = method(current_graph, **args)
            elif func_name == "get_child_node_names": # explicit handling via get_descendant_nodes_data
                if not isinstance(args, dict):
                    raise ValueError(f"'{func_name}' expects a dict argument (e.g., {{'parent_node_id': 'id', ...}}).")
                # assumes signature get_descendant_nodes_data(self, scene_graph, parent_node_id, ...)
                # pass all key-value pairs of the 'args' dict as keyword arguments
                result = method(current_graph, **args)
            elif func_name == "get_edges_for_node":
                if not isinstance(args, dict):
                    raise ValueError(f"'{func_name}' expects a dict argument (e.g., {{'parent_node_id': 'id', ...}}).")
                # assumes signature get_descendant_nodes_data(self, scene_graph, parent_node_id, ...)
                # pass all key-value pairs of the 'args' dict as keyword arguments
                result = method(current_graph, **args)

            elif func_name == "read_node":
                if args is None:
                    raise ValueError("read_node needs at least a node_id.")
                elif isinstance(args, dict):
                    result = method(current_graph, filt=args)
                else:
                    raise ValueError("read_node arguments must be str or dict.")

            else:
                # ⬇︎ Generalization (optional) – match parameters via inspect
                sig = inspect.signature(method)
                kwargs = {}
                for p in list(sig.parameters.values())[1:]:  # first argument = scene_graph
                    if isinstance(args, dict) and p.name in args:
                        kwargs[p.name] = args[p.name]
                result = method(current_graph, **kwargs)

            outcome.update({"execution_successful": True, "result": result})
        except Exception as e:
            outcome["error"] = str(e)

        return outcome
