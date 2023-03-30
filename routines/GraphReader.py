from argparse import ArgumentError
import json
import os
import sys
from copy import deepcopy
sys.path.append('..')
sys.path.append('../simulation')
from dataset_utils import execute_script_utils as utils
from evolving_graph import scripts

from evolving_graph.scripts import Script, parse_script_line
from evolving_graph.environment import EnvironmentGraph, State
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils

scene_num = str(2)
base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
reference_graph_file = base_dir+'/example_graphs/TrimmedTestScene'+scene_num+'_graph.json'
base_graph_file = base_dir+'/example_graphs/CustomBareScene'+scene_num+'_graph.json'
init_graph_file = base_dir+'/example_graphs/CustomScene'+scene_num+'_graph.json'
unnecessary_nodes = ['floor','wall','ceiling','window','character','doorjamb']

def get_object_states(available_states, custom_options):
    object_states = []
    if "closed" in available_states or "open" in available_states:
        if "OPEN" in custom_options:
            object_states.append("OPEN")
        else:
            object_states.append("CLOSED")

    if "off" in available_states or "on" in available_states:
        if "ON" in custom_options:
            object_states.append("ON")
        else:
            object_states.append("OFF")

    if "dirty" in available_states or "clean" in available_states:
        if "DIRTY" in custom_options:
            object_states.append("DIRTY")
        else:
            object_states.append("CLEAN")

    if "plugged" in available_states or "unplugged" in available_states:
        if "PLUGGED_OUT" in custom_options:
            object_states.append("PLUGGED_OUT")
        else:
            object_states.append("PLUGGED_IN")

    return object_states

# def setup():
#     comm = UnityCommunication()
#     return comm

# def process_file_inbuilt(script_file='program.txt', graph_file=init_graph_file):
#     return utils.render_script_from_path(setup(),
#                                 script_file, graph_file,
#                                 {"processing_time_limit": 500, "image_width": 320, "image_height": 240, "image_synthesis": ['normal'], "gen_vid": True, "file_name_prefix": "test", "camera_mode": 'PERSON_TOP'}, scene_id=scene_num)

def remove_nodes_from_graph(graph_file, nodes_to_remove, target_graph_file):
    with open (graph_file,'r') as f:
        graph_dict = json.load(f)
    trimmed_graph = {'nodes':[], 'edges':[]}
    node_ids_to_remove = []
    for node in graph_dict['nodes']:
        if node['class_name'] in nodes_to_remove:
            node_ids_to_remove.append(node['id'])
        else:
            trimmed_graph['nodes'].append(node)
    for edge in graph_dict['edges']:
        if edge['from_id'] in node_ids_to_remove or edge['to_id'] in node_ids_to_remove:
            continue
        trimmed_graph['edges'].append(edge)
    with open (target_graph_file,'w') as f:
        json.dump(trimmed_graph, f, indent=4)

class GraphReader():
    def __init__(self, graph=''):
        if type(graph) == str:
            with open(graph) as f:
                self.graph_dict = json.load(f)
        elif type(graph) == dict:
            self.graph_dict = graph
        else:
            raise ArgumentError(f'GraphReader must be initialized with a string filepath or dict, not {type(graph)}')
        nodes = {n['id']:n['class_name'] for n in self.graph_dict['nodes']}
        nodes_by_room = {n['class_name']:{n['id']:n['class_name']} for n in self.graph_dict['nodes'] if n['category'] == "Rooms"}
        self.node_rooms = {n['id']:n['class_name'] for n in self.graph_dict['nodes'] if n['category'] == "Rooms"}
        self.node_map = {'<'+n['class_name']+'>': '<'+n['class_name']+'> ('+str(n['id'])+')' for n in self.graph_dict['nodes'] if n['category'] == "Rooms"}
        self.usable_nodes = {n['class_name']:(n['id'],n['class_name'],n['class_name']) for n in self.graph_dict['nodes'] if n['category'] == "Rooms"}
        self.node_ids = {}

        remove_keys = []
        # edges = {}
        for e in self.graph_dict['edges']:
            rel = e['relation_type']
            if rel != "CLOSE" and rel!= "FACING":
                n1 = e['from_id']
                n2 = e['to_id']
                if nodes[n1] not in unnecessary_nodes and nodes[n2] not in unnecessary_nodes:
                    # edge_id = (nodes[n1],rel,nodes[n2])
                    # if edge_id not in edges:
                    #     edges[edge_id] = e.update({'from_class':nodes[n1], 'to_class':nodes[n2]})
                    if nodes[n2] in nodes_by_room:
                        add_alias = False
                        if nodes[n1] not in self.usable_nodes:
                            self.usable_nodes[nodes[n1]] = (n1, nodes[n1], nodes[n2])
                        elif self.usable_nodes[nodes[n1]][2] != nodes[n2]:
                            if nodes[n1] not in remove_keys:
                                self.usable_nodes[nodes[n1]+'_'+self.usable_nodes[nodes[n1]][2]] = self.usable_nodes[nodes[n1]]
                                remove_keys.append(nodes[n1])
                            if nodes[n1]+'_'+nodes[n2] not in self.usable_nodes.keys():
                                self.usable_nodes[nodes[n1]+'_'+nodes[n2]] = (n1, nodes[n1], nodes[n2])
                            else:
                                print(self.usable_nodes[nodes[n1]+'_'+nodes[n2]], 'exists!', end = ' ')
                                add_alias = True
                        else:
                            print(self.usable_nodes[nodes[n1]], 'exists!', end = ' ')
                            add_alias = True

                        if add_alias:
                            base = nodes[n1]+'_'+nodes[n2]
                            i = 1
                            alias = base+str(i)
                            while alias in self.usable_nodes.keys():
                                i += 1
                                alias = base+str(i)
                            self.usable_nodes[alias] = (n1, nodes[n1], nodes[n2])
                            # while add_alias:
                            #     alias = input(f'Enter alias for {nodes[n1]} in {nodes[n2]}. Press \'x\' to remove object')
                            #     if alias.lower() == 'x' or alias == '':
                            #         print('Removed')
                            #         break
                            #     elif alias not in self.usable_nodes.keys():
                            #         self.usable_nodes[alias] = (n1, nodes[n1], nodes[n2])
                            #         print('Set to ',alias)
                            #         add_alias = False
                            #     else:
                            #         ow = input('Already exists : {self.usable_nodes[alias]}, Do you want to overwrite? (y/n)')
                            #         if ow.lower() == 'y':
                            #             self.usable_nodes[alias] = (n1, nodes[n1], nodes[n2])
                            #             print('Set to ',alias)
                            #             break

        self.usable_nodes_by_room = {n['class_name']:{} for n in self.graph_dict['nodes'] if n['category'] == "Rooms"}
        self.ref_nodes = {n['class_name']:{} for n in self.graph_dict['nodes'] if n['category'] == "Rooms"}
        for full_name,(id, nodeclass, room) in self.usable_nodes.items():
            self.usable_nodes_by_room[room][full_name] = (nodeclass,id)
            self.ref_nodes[room][nodeclass] = id
            self.node_map[f'<{full_name}>'] = f'<{nodeclass}> ({id})'
            self.node_rooms[id] = room
            self.node_ids[full_name] = id
        
        with open (base_dir+'/resources/object_states.json','r') as f:
            self.object_states = json.load(f)
        with open (base_dir+'/resources/properties_data.json','r') as f:
            self.object_properties = json.load(f)
        self.new_obj_id = 1000
    
    def add(self, obj, relation, parent_id, category="placable_objects", custom_states=[], verbose = False):
        assert(relation in ["INSIDE","ON"])
        if obj in self.object_states.keys():
            object_states = get_object_states(self.object_states[obj], custom_states)
        else:
            object_states = []
            print(f'States not found for {obj}')
        self.graph_dict['nodes'].append({"id": self.new_obj_id, "class_name": obj, "category": category, "properties": self.object_properties[obj], "states": object_states})
        self.graph_dict['edges'].append({"from_id":self.new_obj_id, "relation_type":relation, "to_id":parent_id})
        if verbose:
            print('Adding   : ',self.graph_dict['nodes'][-1])
            print('  Adding : ',self.graph_dict['edges'][-1])
        room_name = None
        for e in self.graph_dict['edges']:
            if e is not None:
                if e['from_id'] == parent_id and e['relation_type'] in ["INSIDE","ON"]:
                    ne = e.copy()
                    ne.update({"from_id":self.new_obj_id, "from_class":obj})
                    self.graph_dict['edges'].append(ne)
                    if verbose:
                        print('  Adding : ',self.graph_dict['edges'][-1])
        self.usable_nodes_by_room[self.node_rooms[parent_id]][obj] = (obj, self.new_obj_id)
        self.ref_nodes[self.node_rooms[parent_id]][obj] = self.new_obj_id
        self.node_rooms[self.new_obj_id] = self.node_rooms[parent_id]
        self.new_obj_id += 1
    
    def write(self, filename):
        with open (filename,'w') as f:
            json.dump(self.graph_dict, f, indent=4)

    def clean_graph(self):
        new_nodes = []
        for n in self.graph_dict['nodes']:
            nc = dict(n)
            if 'bounding_box' in nc:
                del nc['bounding_box']
            new_nodes.append(nc)
        self.graph_dict['nodes'] = new_nodes

    def write_node_map(self, filepath):
        with open('../resources/class_name_equivalence.json') as f:
            name_eq = json.load(f)
        augmented_map = deepcopy(self.node_map)
        for k,v in self.node_map.items():
            for base, eqs in name_eq.items():
                if base in k:
                    for eq in eqs:
                        print(k.replace(base,eq), v)
                        augmented_map[k.replace(base,eq)] = v

        with open(filepath, 'w') as f:
            json.dump(augmented_map, f, indent=4)