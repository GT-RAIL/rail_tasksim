import json
from evolving_graph.scripts import Script, parse_script_line
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils

def read_program(file_name, node_map):
    action_headers = []
    action_scripts = []
    with open(file_name) as f:
        lines = []
        index = 1
        for line in f:
            if line.startswith('##'):
                header = line[2:].strip()
                print(header)
                action_headers.append(header)
                action_scripts.append(lines)
                lines = []
                index = 1
            if '[' not in line:
                continue
            line = line.strip()
            if len(line) > 0 and not line.startswith('#'):
                mapped_line = line
                for full_name, name_id in node_map.items():
                    mapped_line = mapped_line.replace(full_name, name_id)
                scr_line = parse_script_line(mapped_line, index, custom_patt_params = r'\<(.+?)\>\s*\((.+?)\)')
                print(scr_line)
                lines.append(scr_line)
                index += 1
    return action_headers, action_scripts

def execute_program(program_file, graph_file, node_map):
    with open (graph_file,'r') as f:
        init_graph = EnvironmentGraph(json.load(f))
    action_headers, action_scripts = read_program(program_file, node_map)
    name_equivalence = utils.load_name_equivalence()
    executor = ScriptExecutor(init_graph, name_equivalence)
    graphs = []
    for header, script in zip(action_headers, action_scripts):
        success, state, _ = executor.execute(Script(script), w_graph_list=False)
        if not success:
            raise RuntimeError(f'Execution of {header} failed')
        graphs.append(state._graph)
    return action_headers, graphs