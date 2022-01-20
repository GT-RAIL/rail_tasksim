import json
import sys
sys.path.append('..')
from evolving_graph.scripts import Script, parse_script_line
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils

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

    for n in nodes_removed:
        print ('Removed node : ',n)
    for n in nodes_added:
        print ('Added node   : ',n)
    for e in edges_removed:
        c1 = class_from_id(g1,e['from_id'])
        c2 = class_from_id(g1,e['to_id'])
        if c1 != 'character' and c2 != 'character' and e['relation_type'] in ['INSIDE','ON']:
            print ('Removed edge : ',c1,e['relation_type'],c2)
    for e in edges_added:
        c1 = class_from_id(g2,e['from_id'])
        c2 = class_from_id(g2,e['to_id'])
        if c1 != 'character' and c2 != 'character' and e['relation_type'] in ['INSIDE','ON']:
            print ('Added edge   : ',c1,e['relation_type'],c2)

def read_program(file_name, node_map):
    action_headers = []
    action_scripts = []
    action_objects_in_use = []

    def obj_class_id_from_string(string_in):
        class_id = [a[1:-1] for a in string_in.split(' ')]
        return (int(class_id[1]), class_id[0])

    with open(file_name) as f:
        lines = []
        full_program = []
        obj_start, obj_end = [], []
        index = 1
        object_use = {'start':[], 'end':[]}
        for line in f:
            if line.startswith('##'):
                header = line[2:].strip()
                action_headers.append(header)
                action_scripts.append(lines)
                object_use['start'].append(obj_start)
                object_use['end'].append(obj_end)
                lines = []
                obj_start, obj_end = [], []
                index = 1
            line = line.strip()
            if line.startswith('+'):
                obj = obj_class_id_from_string(node_map[line[1:]])
                obj_start.append(obj)
                continue
            if line.startswith('-'):
                obj = obj_class_id_from_string(node_map[line[1:]])
                obj_end.append(obj)
                continue
            if '[' not in line:
                continue
            if len(line) > 0 and not line.startswith('#'):
                mapped_line = line
                for full_name, name_id in node_map.items():
                    mapped_line = mapped_line.replace(full_name, name_id)
                # print(line,' -> ', mapped_line)
                scr_line = parse_script_line(mapped_line, index, custom_patt_params = r'\<(.+?)\>\s*\((.+?)\)')
                lines.append(scr_line)
                full_program.append(scr_line)
                index += 1
        action_scripts.append(lines)
        action_scripts = action_scripts[1:]
    return action_headers, action_scripts, object_use, full_program

def execute_program(program_file, graph_file, node_map):
    with open (graph_file,'r') as f:
        init_graph = EnvironmentGraph(json.load(f))
    action_headers, action_scripts, action_obj_use, whole_program = read_program(program_file, node_map)
    name_equivalence = utils.load_name_equivalence()
    graphs = [init_graph.to_dict()]
    print('Checking scripts...',end='')
    for script in action_scripts:
        executor = ScriptExecutor(EnvironmentGraph(graphs[-1]), name_equivalence)
        success, state, graph_list = executor.execute(Script(script), w_graph_list=True)
        if not success:
            script_string = '\n  - '.join([str(l) for l in script])
            raise RuntimeError(f'Execution of the following script failed because {executor.info.get_error_string()} \n  - {script_string}')
        graphs.append(state.to_dict())
        # print([str(l) for l in script])
        # print_graph_difference(graphs[-2],graphs[-1])
        # input('Press something...')
    
    print("Script segments OK")
    executor = ScriptExecutor(EnvironmentGraph(graphs[-1]), name_equivalence)
    print('Checking final state...',end='')
    executor.check_final_state()
    print('Final state OK\n')
    try:
        print('Trying a second run...',end='')
        success, _, _ = executor.execute(Script(whole_program), w_graph_list=True)
        if not success:
            raise RuntimeError(f'Execution failed because {executor.info.get_error_string()}')
        print("Execution successful!!")
    except Exception as e: print (e)

    return action_headers, graphs