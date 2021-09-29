import os
import re
import json

import sys
sys.path.append('simulation')
from unity_simulator.comm_unity import UnityCommunication
from dataset_utils import execute_script_utils as utils
from evolving_graph import scripts

import pdb

def parse_exec_script_file(file_name):
    with open(file_name, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        title = content[0]
        description = content[1]
        script_raw = content[4:]

    script = []
    for elem in script_raw:
        script.append(re.sub('[0-9]\.', '', elem))
    print(script)
    return title, description, script

def obtain_objects_from_message(message):
    objects_missing = []
    for x in ['unplaced', 'missing_destinations', 'missing_prefabs']:
        if x in message.keys():
            objects_missing += message[x]
    return objects_missing


def setup():
    comm = UnityCommunication()
    return comm

SCRIPT_FILE = "dataset/programs_processed/executable_programs/TrimmedTestScene1_graph/results_intentions_march-13-18/file3_1.txt"
GRAPH_FILE = SCRIPT_FILE.replace('executable_programs','init_and_final_graphs')
GRAPH_FILE = GRAPH_FILE.replace('.txt','.json')

if __name__ == "__main__":
    # # OPTION ONE: use existing code
    if (False):
        utils.render_script_from_path(setup(), 
                                  SCRIPT_FILE,
                                  GRAPH_FILE,
                                  {"processing_time_limit": 120, "image_width": 320, "image_height": 240, "image_synthesis": ['normal'], "gen_vid": True, "file_name_prefix": "test"})

    # OPTION TWO: what the above runs "under the hood"
    if (True):
        # get comm
        comm = setup()
        status = comm.reset(0)
        if not status:
            print("Error: could not load sim.")
            exit()
        # get script
        try:
            title, desc, script = parse_exec_script_file(SCRIPT_FILE)
        except FileNotFoundError:
            print("Error: could not load script file.")
            exit()
        script_content = scripts.read_script_from_list_string(script)
        print("TITLE: " + title)
        print("DESCRIPTION: " + desc)
        print("### SCRIPT ###")
        print(script)
        print("### SCRIPT ###")
        # get init_graph
        graphs = None
        try:
            with open(GRAPH_FILE, "r") as f:
                graphs = json.load(f)
        except FileNotFoundError:
            print("Error: could not load graph file.")
            exit()
        init_graph = graphs["init_graph"]
        status, msg = comm.expand_scene(init_graph)
        if not status:
            # sometimes fails due to "background" not renderable by unity
            objects_missing = obtain_objects_from_message(msg)
            objects_script = [x[0].replace('_', '') for x in script_content.obtain_objects()]
            intersection_objects = list(set(objects_script).intersection(objects_missing))
            if len(intersection_objects) > 0:
                print("Error: could not load init_graph. See below.")
                print(msg)
                exit()
        # finally with comm, graph, and script, render the script
        render_args = {"processing_time_limit": 120,
                    "image_width": 320,
                    "image_height": 240,
                    "image_synthesis": ['normal'],
                    "gen_vid": True,
                    "file_name_prefix": "test"}
        status, msg = comm.render_script(script, **render_args)
        if status:
            print("robotHow recipe executed!")
        else:
            print(msg)