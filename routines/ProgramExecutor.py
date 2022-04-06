import json
import sys
sys.path.append('..')
from evolving_graph.scripts import Script, parse_script_line
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils
from object_locations import object_locations

def class_from_id(graph, id):
    lis = [n['class_name'] for n in graph['nodes'] if n['id']==id]
    if len(lis) > 0:
        return lis[0]
    else:
        return 'None'

def print_graph_difference(g1,g2):
    edges_removed = [e for e in g1['edges'] if e not in g2['edges']]
    edges_added = [e for e in g2['edges'] if e not in g1['edges']]
    nodes_removed = [n for n in g1['nodes'] if n['id'] not in [n2['id'] for n2 in g2['nodes']]]
    nodes_added = [n for n in g2['nodes'] if n['id'] not in [n2['id'] for n2 in g1['nodes']]]
    ignore_for_edges = ['wall']

    for n in nodes_removed:
        print ('Removed node : ',n)
    for n in nodes_added:
        print ('Added node   : ',n)
    remaining_objects = []
    for e in edges_removed:
        c1 = class_from_id(g1,e['from_id'])
        c2 = class_from_id(g1,e['to_id'])
        if c1 not in ignore_for_edges and c2 not in ignore_for_edges and e['relation_type'] in ['INSIDE','ON','HOLDS_RH','HOLDS_LH']:
            print (' - ',c1,e['relation_type'],c2)
            remaining_objects.append(e['from_id'])
    for e in edges_added:
        c1 = class_from_id(g2,e['from_id'])
        c2 = class_from_id(g2,e['to_id'])
        if c1 not in ignore_for_edges and c2 not in ignore_for_edges and e['relation_type'] in ['INSIDE','ON','HOLDS_RH','HOLDS_LH']:
            print (' + ',c1,e['relation_type'],c2)
            if e['from_id'] in remaining_objects:
                remaining_objects.remove(e['from_id'])
    # for id in remaining_objects:
    #     for e in g2['edges']:
    #         if e['from_id'] == id and e['relation_type'] in ['INSIDE','ON']:
    #             c2 = class_from_id(g2,e['to_id'])
    #             if c2 not in ignore_for_edges:
    #                 c1 = class_from_id(g2,e['from_id'])
    #                 print (' + ',c1,e['relation_type'],c2)


def read_program(file_name, node_map):
    action_headers = []
    action_scripts = []
    action_objects_in_use = []

    def obj_class_id_from_string(string_in):
        class_id = [a[1:-1] for a in string_in.split(' ')]
        return (int(class_id[1]), class_id[0])

    def get_duration(header):
        durations = (header).split('-')
        assert len(durations)==2, f"Invalid time range {header} in {file_name}"
        duration_min = int(durations[0].strip())
        duration_max = int(durations[1].strip())
        return duration_min, duration_max

    with open(file_name) as f:
        lines = []
        durations = []
        obj_start, obj_end = [], []
        index = 1
        for line in f:
            if line.startswith('##'):
                num_lines = len(lines) - len(durations)
                if num_lines > 0:
                    durations += [((duration_min/num_lines),(duration_max/num_lines))] * num_lines
                header = line[2:].strip()
                duration_min, duration_max = get_duration(header)
            line = line.strip()
            if line.startswith('+'):
                obj = obj_class_id_from_string(node_map[line[1:]])
                obj_start[-1].append(obj)
                continue
            if line.startswith('-'):
                obj = obj_class_id_from_string(node_map[line[1:]])
                obj_end[-1].append(obj)
                continue
            if '[' not in line:
                continue
            if len(line) > 0 and not line.startswith('#'):
                mapped_line = line
                for full_name, name_id in node_map.items():
                    mapped_line = mapped_line.replace(full_name, name_id)
                # print(line,' -> ', mapped_line)
                try:
                    scr_line = parse_script_line(mapped_line, index, custom_patt_params = r'\<(.+?)\>\s*\((.+?)\)')
                except Exception as e:
                    print(f'The following line has a mistake! Did you write the correct object and activity names? \n {line}')
                    raise e
                lines.append(scr_line)
                obj_start.append([])
                obj_end.append([])
                index += 1
        num_lines = len(lines) - len(durations)
        if num_lines > 0:
            durations += [((duration_min/num_lines),(duration_max/num_lines))] * num_lines
    return durations, lines, obj_start, obj_end

def execute_program(program_file, graph_file, node_map, verbose=False):
    with open (graph_file,'r') as f:
        init_graph_dict = json.load(f)
    init_graph = EnvironmentGraph(init_graph_dict)
    durations, lines, obj_start, obj_end = read_program(program_file, node_map)
    name_equivalence = utils.load_name_equivalence()
    graphs = [init_graph.to_dict()]

    print('Checking scripts...',end='')
    executor = ScriptExecutor(EnvironmentGraph(graphs[-1]), name_equivalence)
    success, state, graph_list = executor.execute(Script(lines), w_graph_list=True)
    print('exec info ---- ')
    print(executor.info.get_error_string())
    if not success:
        error_str = executor.info.get_error_string()
        if 'inside other closed thing' in error_str:
            object = error_str[error_str.index('<')+1:error_str.index('>')]
            print(f'{object} is inside {object_locations[object]}')
        raise RuntimeError(f'Execution failed because {error_str}')
    print('Execution successful!!')

    print('Checking final state...',end='')
    executor = ScriptExecutor(EnvironmentGraph(state.to_dict()), name_equivalence)
    executor.check_final_state()
    print('Final state OK\n')

    if verbose:
        print('This is how the scene changes after every set of actions...')
        for script, graph, prev_graph in zip(lines,graph_list[1:],graph_list[:-1]):
            print('Changes from ',script)
            if verbose:
                print_graph_difference(prev_graph,graph)
    