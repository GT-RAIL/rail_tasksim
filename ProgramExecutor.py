import json
from evolving_graph.scripts import Script, parse_script_line
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils


class ProgramExecutor():
    def __init__(self, program_file, graph_file, node_map):
        with open (graph_file,'r') as f:
            self.init_graph_dict = json.load(f)
        self.init_graph = EnvironmentGraph(self.init_graph_dict)
        self.node_map = node_map
        self.read_program(program_file)

    def read_program(self, file_name):
        self.action_scripts = []
        self.single_script = []
        self.action_headers = []
        with open(file_name) as f:
            lines = []
            index = 1
            for line in f:
                if line.startswith('##'):
                    header = line[2:].strip()
                    print(header)
                    self.action_headers.append(header)
                    self.action_scripts.append(lines)
                    lines = []
                    index = 1
                if '[' not in line:
                    continue
                line = line.strip()
                if len(line) > 0 and not line.startswith('#'):
                    mapped_line = line
                    for full_name, name_id in self.node_map.items():
                        mapped_line = mapped_line.replace(full_name, name_id)
                    scr_line = parse_script_line(mapped_line, index, custom_patt_params = r'\<(.+?)\>\s*\((.+?)\)')
                    print(scr_line)
                    self.single_script.append(scr_line)
                    lines.append(scr_line)
                    index += 1

    def execute(self):
        name_equivalence = utils.load_name_equivalence()
        executor = ScriptExecutor(self.init_graph, name_equivalence)
        graphs = []
        for header, script in zip(self.action_headers, self.action_scripts):
            success, state, _ = executor.execute(Script(script), w_graph_list=False)
            if not success:
                raise RuntimeError(f'Execution of {header} failed')
            graphs.append(state._graph)
        return self.action_headers, graphs