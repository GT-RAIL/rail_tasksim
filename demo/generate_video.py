# Generate video for a program. Make sure you have the executable open
import sys
sys.path.append('../simulation/')
from os import path

from unity_simulator.comm_unity import UnityCommunication
from evolving_graph.utils import load_graph

import random
import json
# activity_paths = json.load(open('../dataset/activity_paths.json'))
# SCRIPT_FILE = random.sample(random.choice(list(activity_paths.values())),1)[0]['program']
# SCRIPT_FILE = 'example_scripts/example_script_3.txt'
SCRIPT_FILE = '../dataset/programs_processed/executable_programs/TrimmedTestScene4_graph/results_text_rebuttal_specialparsed_programs_turk_july/split19_3.txt'
import copy
graph_file = copy.deepcopy(SCRIPT_FILE)
graph_file = graph_file.replace('executable_programs','init_and_final_graphs')
graph_file = graph_file.replace('.txt','.json')
with open(graph_file, 'r') as f:
	graphs = json.load(f)
	first_graph = graphs['init_graph']

def read_script_file(file_name):
    script_list = []
    with open(file_name) as f:
        for line in f:
            if '[' in line:
                print(line)
                script_list.append(line.strip())
    return script_list

script = read_script_file(SCRIPT_FILE) # Add here your script
folder_name = 'try' #path.basename(path.splitext(SCRIPT_FILE)[0])
print('Starting Unity...')
comm = UnityCommunication()
print('Starting scene...')
comm.reset(3)
comm.expand_scene(first_graph)
print('Generating video...')
success, message = comm.render_script(script, processing_time_limit=200, file_name_prefix=folder_name, save_pose_data=False, gen_vid=True, save_scene_states=True, character_resource='Chars/Male1', camera_mode='PERSON_TOP')
if message is not None:
    print(message)
if success:
    print('Generated, find video in Output/',folder_name)
else:
    print('Trouble rendering the script!')
