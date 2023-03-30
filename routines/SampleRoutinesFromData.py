# ## Sample scripts for full routines from sourced scripts and sourced schedules

import json
import os
import queue
import shutil
import argparse
from math import ceil
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import sys

sys.path.append('..')
sys.path.append('../simulation')

from evolving_graph.scripts import Script
import evolving_graph.utils as utils
from evolving_graph.execution import ScriptExecutor
from evolving_graph.environment import EnvironmentGraph

from GraphReader import GraphReader, init_graph_file, scene_num
from ProgramExecutor import read_program
from ScheduleDistributionSampler import ScheduleDistributionSampler, persona_options, individual_options, ScheduleSampler_FixedSequence
from ScheduleSamplerNoisy import ScheduleSamplerNoisy, ScheduleSamplerThreeActivities, ScheduleSamplerBreakfast, ScheduleSamplerBreakfastOrLeave, ScheduleSamplerPhoneInterruption

# set_seed = 23424
# random.seed(set_seed)
# np.random.seed(set_sseed)

def time_mins(mins, hrs, day=0):
    return (((day)*24)+hrs)*60+mins

def time_human(time_mins):
    time_mins = int(round(time_mins))
    mins = time_mins%60
    time_mins = time_mins//60
    hrs = time_mins%24
    time_mins = time_mins//24
    days = time_mins
    h = '{:02d}:{:02d}'.format(hrs,mins)
    if days != 0:
        h = str(days)+'day - '+h
    return h

def check_routine_conditions(activities_info_list, med=False):
    return 
    activities_list = [a[0] for a in activities_info_list]
    if len(set(activities_list)) < 2:
        raise SamplingFailure(f'{activities_list} are not enough activities')
    # return
    if 'brushing_teeth' not in activities_list:
        raise SamplingFailure('Brushing teeth must happen at least once')    
    first_brush_start_time = min([a[1] for a in activities_info_list if a[0]=='brushing_teeth'])
    last_brush_start_time = max([a[1] for a in activities_info_list if a[0]=='brushing_teeth'])
    breakfast_start_time = [a[1] for a in activities_info_list if a[0]=='breakfast']
    dinner_start_time = [a[1] for a in activities_info_list if a[0]=='dinner']
    if 'breakfast' in activities_list and first_brush_start_time - breakfast_start_time[0] < 6*60 and first_brush_start_time - breakfast_start_time[0] > 0:
        raise SamplingFailure(f'Brushing teeth at {time_human(first_brush_start_time)} must happen before breakfast{time_human(breakfast_start_time[0])}!')
    if 'dinner' in activities_list and dinner_start_time[0] - last_brush_start_time < 6*60 and dinner_start_time[0] - last_brush_start_time > 0:
        raise SamplingFailure(f'Brushing teeth at {time_human(last_brush_start_time )} must happen after dinner{time_human(dinner_start_time[0])}!')
    if med and not 'taking_medication' in activities_list:
        raise SamplingFailure('Medecines must be taken')
    time_away = 0
    if 'leave_home' in activities_list:
        time_away_1 = [a[1] for a in activities_info_list if a[0]=='leave_home']
        time_away_1.sort()
        time_away_2 = [a[1] for a in activities_info_list if a[0]=='come_home']
        time_away_2.sort()
        time_away = sum([a2-a1 for a1,a2 in zip(time_away_1, time_away_2)])
    if len([a for a in activities_list if a in ['breakfast','lunch','dinner']]) < 2 and time_away < 1:
        raise SamplingFailure(f'Not enough meals in {activities_list}')
    computer_work_time = sum([a[2]-a[1] for a in activities_info_list if a[0]=='computer_work'])
    if computer_work_time > 10:
        raise SamplingFailure('Computer work exceeds 10 hours!')
    if activities_list.count('vaccuum_cleaning') > 1:
        raise SamplingFailure(f'Vaccuum Claeaning can only happen once a day not {activities_list}')

info = {}
info['dt'] = 10   # minutes
info['num_train_routines'] = 50
info['num_test_routines'] = 10
info['start_time'] = time_mins(mins=0, hrs=6)
info['end_time'] = time_mins(mins=0, hrs=24)
info['only_used_objects'] = True
info['idle_sampling_factor'] = 1.0
info['block_activity_for_hrs'] = 5
info['interruption_probability'] = 0.0
info['different_activity_scripts'] = False

init_graph = GraphReader(init_graph_file)
print(f'Using scene {int(scene_num)-1}, i.e. \'TestScene{scene_num}\'')
ignore_classes = ['floor','wall','ceiling','character']
utilized_object_ids = set()
edge_classes = ["INSIDE", "ON"]
all_activities = set()
all_activities_coarse = set()

with open ('data/personaBasedSchedules/transitions_best.json') as f:
    ideal_transitions = json.load(f)
    
class SamplingFailure(Exception):
    pass

def choose_script(list_of_scripts):
    if info['different_activity_scripts']:
        return random.choice(list_of_scripts)
    else:
        return list_of_scripts[0]

def get_scripts(n):
    scripts_list = {}
    directory = os.path.join('data/sourcedScriptsByActivity')
    for activity in os.listdir(directory):
        available_files = os.listdir(os.path.join(directory,activity))
        if not available_files:
            continue
        available_files.sort()
        nact = deepcopy(n) 
        if nact < 0:
            scripts_list[activity] = [{'filename': f} for f in available_files]
        else:
            while nact >= len(available_files):
                nact -= len(available_files)
            f = available_files[nact]
            scripts_list[activity] = [{'filename': f}]
    scripts_list["come_home"] = deepcopy(scripts_list["leave_home"])
    print(init_graph.node_map)
    for activity, info_list in scripts_list.items():
        for idx, info in enumerate(info_list):
            info = read_program(os.path.join('data/sourcedScriptsByActivity',activity,info['filename']), init_graph.node_map)
            scripts_list[activity][idx].update(info)
    return scripts_list

# %% Schedule

# class Schedule():
class ScheduleWithInterruptions():
    def __init__(self, sampler_name, scripts_list, num_optional_activities=-1, output_dir=None, sampler_specification=None):
        global info
        # sampler = ScheduleSampler_FixedSequence()
        # sampler = ScheduleDistributionSampler(type=sampler_name, idle_sampling_factor=info['idle_sampling_factor'], resample_after=info['block_activity_for_hrs'], num_optional_activities=num_optional_activities)
        if sampler_name == 'breakfast':
            sampler = ScheduleSamplerBreakfast()
        elif sampler_name == 'scripted':
            sampler = ScheduleSamplerNoisy()
        elif sampler_name == 'breakfast_or_leave':
            sampler = ScheduleSamplerBreakfastOrLeave()
        elif sampler_name == 'phone_interruption':
            sampler = ScheduleSamplerPhoneInterruption()
        elif sampler_name == 'two_scenarios_simple':
            sampler = ScheduleSamplerThreeActivities()
        else:
            sampler = ScheduleDistributionSampler(type=sampler_name, idle_sampling_factor=info['idle_sampling_factor'], resample_after=info['block_activity_for_hrs'], num_optional_activities=num_optional_activities, specification=sampler_specification)
            # raise Exception(f'Unrecognized sampler {sampler}')
        self.all_actions = []
        self.script_header = ''
        t = info['start_time']
        act_start_time = t
        activities_list = []
        while t < info['end_time']:
            activity_name = sampler(t)
            act_start_time = t
            if activity_name is None:
                t += info['dt']
                continue
            new_actions = []
            activity_info = choose_script(scripts_list[activity_name])
            sampler.update_distributions(t, deepcopy(activity_name))
            for line, duration, act_label in zip(activity_info['lines'], activity_info['durations'], activity_info['activity_labels']):
                d = (random.random() * (duration[1] - duration[0]) + duration[0])
                new_actions.append((line, d, act_label, activity_name, activity_info['filename']))
            changepoints = np.argwhere([a1[2]!=a2[2] for a1,a2 in zip(new_actions[1:], new_actions[:-1])])[:,0]
            interrupted = random.random() < info['interruption_probability']
            changeindex = -1
            if interrupted and len(changepoints)>0:
                changeindex = random.choice(changepoints)
            if interrupted:
                for line, d, act_label, act_name, act_scr in new_actions[:changeindex+1]:
                    self.all_actions.append({'script':line, 'time_from':t, 'time_to':t+d, 'time_from_h':time_human(t), 'time_to_h':time_human(t+d), 'name':act_name+'-'+act_scr, 'activity_label':act_label, 'activity_name':act_name, 'activity_label_coarse':act_name})
                    t+=d
                interruption_activity_name = sampler(t, interruption=True)
                if interruption_activity_name is None:
                    interrupted = False
                else:
                    interruption_activity_info = choose_script(scripts_list[interruption_activity_name])
                    interruption_time_start = t
                    for line, duration, act_label in zip(interruption_activity_info['lines'], interruption_activity_info['durations'], interruption_activity_info['activity_labels']):
                        d = (random.random() * (duration[1] - duration[0]) + duration[0])
                        self.all_actions.append({'script':line, 'time_from':t, 'time_to':t+d, 'time_from_h':time_human(t), 'time_to_h':time_human(t+d), 'name':interruption_activity_name+'-'+interruption_activity_info['filename'], 'activity_label':act_label, 'activity_name':interruption_activity_name, 'activity_label_coarse':interruption_activity_name})
                        t+=d
                    interruption_time_end = t
            for line, d, act_label, act_name, act_scr in new_actions[changeindex+1:]:
                self.all_actions.append({'script':line, 'time_from':t, 'time_to':t+d, 'time_from_h':time_human(t), 'time_to_h':time_human(t+d), 'name':act_name+'-'+act_scr, 'activity_label':act_label, 'activity_name':act_name, 'activity_label_coarse':act_name})
                t+=d
            activities_list.append((activity_name, act_start_time, t))
            self.script_header += '{} ({} - {}) \n'.format(activity_name, time_human(act_start_time), time_human(t))
            if interrupted:
                self.script_header += '{} ({} - {}) \n'.format(interruption_activity_name, time_human(interruption_time_start), time_human(interruption_time_end))
        check_routine_conditions(activities_list, med=sampler.does('take_medication'))
        sampler.valid_end(info['end_time'], failure=SamplingFailure)

    def get_combined_script(self, verbose=False):
        return self.all_actions, self.script_header


class ScheduleFromHybridDuration():
    def __init__(self, sampler_name, scripts_list, num_optional_activities=-1):
        global info
        # sampler = ScheduleSampler_FixedSequence()
        sampler = ScheduleDistributionSampler(type=sampler_name, idle_sampling_factor=info['idle_sampling_factor'], resample_after=info['block_activity_for_hrs'], num_optional_activities=num_optional_activities)
        if sampler_name in ideal_transitions.keys():
            self.ideal_transitions = ideal_transitions[sampler_name]
        else:
            self.ideal_transitions = {a:None for a in sampler.activities+['come_home']}
        self.activities = []
        t = info['start_time']
        activity_name = None

        while t < info['end_time']:
            new_activity = sampler(t)
            if new_activity is None:
                if activity_name is not None:
                    self.activities[-1][2]['end_time'] = deepcopy(t)
                    sampler.update_distributions(t, deepcopy(activity_name))
                    activity_name = deepcopy(new_activity)
                    duration_min, duration_max = float("inf"), float("inf")
                t += info['dt']
                continue
            if activity_name == new_activity:
                duration_so_far = t - self.activities[-1][0]
                t += deepcopy(int(random.random() * (duration_max - duration_so_far)))
                continue
            if len(self.activities) > 0 and activity_name is not None:
                self.activities[-1][2]['end_time'] = deepcopy(t)
            sampler.update_distributions(t, deepcopy(activity_name))
            activity_name = deepcopy(new_activity)
            activity_info = choose_script(scripts_list[activity_name])
            self.activities.append((deepcopy(t), deepcopy(activity_name), deepcopy(activity_info)))
            duration_min, duration_max = activity_info['total_duration_range']
            sample_duration = random.random() * (duration_max - duration_min) + duration_min
            t += deepcopy(sample_duration)
        sampler.valid_end(info['end_time'], failure=SamplingFailure)
        if activity_name is not None:
            self.activities[-1][2]['end_time'] = deepcopy(t)

    def get_combined_script(self, verbose=False):
        all_actions = []
        script_header = ''
        for t, activity, info in self.activities:
            start_time = t
            end_time = info['end_time']
            script_header += '{} ({} - {}) \n'.format(activity, time_human(start_time), time_human(end_time))
            remaining_min, remaining_max = deepcopy(info['total_duration_range'])
            duration_remaining = deepcopy(end_time - start_time)
            for act_line, act_duration, act_label in zip(info['lines'], info['durations'], info['activity_labels']):
                remaining_min -= act_duration[0]
                remaining_max -= act_duration[1]
                sampling_min = max(act_duration[0], duration_remaining - remaining_max)
                sampling_max = min(act_duration[1], duration_remaining - remaining_min)
                d = (random.random() * (sampling_max-sampling_min) + sampling_min)
                duration_remaining -= d
                all_actions.append({'script':act_line, 'time_from':t, 'time_to':t+d, 'time_from_h':time_human(t), 'time_to_h':time_human(t+d), 'name':activity+'-'+info['filename'], 'activity_label':act_label, 'activity_label_coarse':activity})
                t += d
        if verbose:
            for a in all_actions:
                print (a['time_from_h']+' to '+a['time_to_h']+' : '+a['name']+' : '+a['script'])

        return all_actions, script_header


# %% run and get graphs

def get_graphs(all_actions, script_string = '',  verbose=False):
    with open (init_graph_file,'r') as f:
        init_graph_dict = json.load(f)
    name_equivalence = utils.load_name_equivalence()
    complete_script_lines = [action['script'] for action in all_actions]
    executor = ScriptExecutor(EnvironmentGraph(init_graph_dict), name_equivalence)
    success, _, graph_list = executor.execute(Script(complete_script_lines), w_graph_list=True)
    if not success:
        script_info = ''
        try:
            for n in range(len(graph_list)-3, len(graph_list)+1):
                script_info += ('\n## Executing '+all_actions[n]['name']+' from '+all_actions[n]['time_from_h']+' to '+all_actions[n]['time_to_h']+' using '+str(all_actions[n]['script']))
        except:
            pass
        raise SamplingFailure('Execution of following failed because {}... {}'.format(executor.info.get_error_string(), script_info))
    
    graphs = [EnvironmentGraph(init_graph_dict).to_dict()]
    times = [info['start_time']]
    activities = []
    activities_coarse = []
    important_objects = set()

    last_source = None
    last_activity = None
    for graph, action_info in zip(graph_list[1:], all_actions):
        src = action_info['name']
        if action_info['name'] != last_source:
            script_string += f'\n\n### {src}\n'
            last_source = src
        all_rel = [edge['relation_type'] for edge in graph['edges']]
        if 'HOLDS_RH' in all_rel or 'HOLDS_LH' in all_rel:
            script_string += '\n' + str(action_info['script'])
            script_string += '\n' + str(action_info['time_to'])
            continue
        if action_info['activity_label'] != last_activity and action_info['time_from']>times[-1]:
            if len(graphs) > 1:
                script_string += "<<Activity Change>>"
            graphs.append(graphs[-1])
            activities.append(None)
            activities_coarse.append(None)
            times.append(action_info['time_from'])
            script_string += '\n## Idle until {}\n'.format(action_info['time_from_h'])
        
        # if len(times) > 0 and action_info['time_to']-times[-1] < 1:
            # graphs[-1] = graph
        else:
            difference = 'first_graph'
            script_string += '\n' + str(action_info['script'])
            script_string += '\n' + str(action_info['time_to'])
            if len(graphs) > 1:
                difference, nodes_moved, nodes_moved_sensored = print_graph_difference(graphs[-1], graph)
                if difference != '':
                    script_string += '\n' + "<<Graph Changes>>"
                    script_string += difference + '\n'
                    if len(nodes_moved) > 0:
                        script_string += '\n' + str(action_info['time_to']) + ': ' + ','.join(nodes_moved) + ' moved'
                    if len(nodes_moved_sensored) > 0:
                        script_string += '\n' + str(action_info['time_to']) + ': ' + ','.join(nodes_moved_sensored) + ' moved_under_sensor'
            if difference == '' and action_info['activity_label'] == activities[-1]:
                times[-1] = action_info['time_to']
            else:
                graphs.append(graph)
                activities.append(action_info['activity_label'])
                activities_coarse.append(action_info['activity_label_coarse'])
                times.append(action_info['time_to'])
        script_string += '\n## {} until {}\n'.format(action_info['activity_label'], action_info['time_to_h'])
        important_objects.update(get_used_objects(graphs[-2],graphs[-1]))
        all_activities.add(action_info['activity_label'])
        all_activities_coarse.add(action_info['activity_label_coarse'])
        last_activity = action_info['activity_label']

    return graphs, times, activities, activities_coarse, script_string, list(important_objects)


# %% Post processing

def remove_ignored_classes(graphs, utilized_object_ids):
    clipped_graphs = []
    for graph in graphs:
        clipped_graphs.append({'nodes':[],'edges':[]})
        ignore_ids = []
        for node in graph['nodes']:
            if node['class_name'] in ignore_classes:
                ignore_ids.append(node['id'])
            elif info['only_used_objects'] and node['id'] not in utilized_object_ids:
                ignore_ids.append(node['id'])
            else:
                clipped_graphs[-1]['nodes'].append(node)
        for edge in graph['edges']:
            if edge['from_id'] in ignore_ids or edge['to_id'] in ignore_ids:
                continue
            clipped_graphs[-1]['edges'].append(edge)
    return clipped_graphs

def class_from_id(graph, id):
    lis = [n['class_name'] for n in graph['nodes'] if n['id']==id]
    if len(lis) > 0:
        return lis[0]
    else:
        return 'None'

def print_graph_difference(g1,g2):
    nodes_moved_to = set()
    nodes_moved_from = set()
    nodes_moved_from_sensored_containers = set()
    edges_removed = [e for e in g1['edges'] if e not in g2['edges']]
    edges_added = [e for e in g2['edges'] if e not in g1['edges']]
    nodes_removed = [n for n in g1['nodes'] if n['id'] not in [n2['id'] for n2 in g2['nodes']]]
    nodes_added = [n for n in g2['nodes'] if n['id'] not in [n2['id'] for n2 in g1['nodes']]]
    state_changes = []
    for n1 in g1['nodes']:
       for n2 in g2['nodes']:
           if n1['id'] == n2['id']:
               if n1['states'] != n2['states']:
                   state_changes.append((n1['id'], n1['class_name'],n1['states'],n2['states']))

    info_str = ''
    sensored_containers = ['']

    for n in nodes_removed:
        info_str += ('\n   - '+str(n))
    for n in nodes_added:
        info_str += ('\n   + '+str(n))
    for sc in state_changes:
        info_str += ('\n   (s)'+str(sc[0])+str(sc[1])+str(sc[2])+'->'+str(sc[3]))
    for e in edges_removed:
        c1 = class_from_id(g1,e['from_id'])
        c2 = class_from_id(g1,e['to_id'])
        if c1 != 'character' and c2 not in ['character', 'floor', 'wall'] and e['relation_type'] in edge_classes:
            info_str += ('\n   - '+c1+' '+e['relation_type']+' '+c2)
            nodes_moved_from.add(c1)
            if c2 in sensored_containers:
                nodes_moved_from_sensored_containers.add(c1)
    for e in edges_added:
        c1 = class_from_id(g2,e['from_id'])
        c2 = class_from_id(g2,e['to_id'])
        if c1 != 'character' and c2 not in ['character', 'floor', 'wall'] and e['relation_type'] in edge_classes:
            info_str += ('\n   + '+c1+' '+e['relation_type']+' '+c2)
            nodes_moved_to.add(c1)
            if c2 in sensored_containers:
                nodes_moved_from_sensored_containers.add(c1)

    nodes_moved = nodes_moved_from.intersection(nodes_moved_to)
    nodes_moved_from_sensored_containers = nodes_moved_from_sensored_containers.intersection(nodes_moved)

    return info_str, list(nodes_moved), list(nodes_moved_from_sensored_containers)

def get_used_objects(g1,g2):
    utilized_object_ids = []
    edges_removed = [e for e in g1['edges'] if e not in g2['edges']]
    edges_added = [e for e in g2['edges'] if e not in g1['edges']]
    nodes_removed = [n for n in g1['nodes'] if n['id'] not in [n2['id'] for n2 in g2['nodes']]]
    nodes_added = [n for n in g2['nodes'] if n['id'] not in [n2['id'] for n2 in g1['nodes']]]
    for n in nodes_removed:
        utilized_object_ids.append(n['id'])
    for n in nodes_added:
        utilized_object_ids.append(n['id'])
    for e in edges_removed:
        if e['relation_type'] in edge_classes and class_from_id(g1,e['from_id'])!='character' and class_from_id(g1,e['to_id'])!='character':
            utilized_object_ids.append(e['from_id'])
            utilized_object_ids.append(e['to_id'])
    for e in edges_added:
        if e['relation_type'] in edge_classes and class_from_id(g1,e['from_id'])!='character' and class_from_id(g1,e['to_id'])!='character':
            utilized_object_ids.append(e['from_id'])
            utilized_object_ids.append(e['to_id'])
    return utilized_object_ids

# %% Make a routine
def make_routine(routine_num, scripts_dir, routines_dir, sampler_name, scripts_list, logs_dir, misc_output_dir=None, clean_data=False, verbose=False, sampler_specification=None):
    global info
    print('.',end='')
    while True:
        try:
            # if sampler_name in ['scripted','breakfast','breakfast_or_leave','phone_interruption']:
            s = ScheduleWithInterruptions(sampler_name, scripts_list, output_dir=misc_output_dir, sampler_specification=sampler_specification)
            # else:
            #     s = ScheduleFromHybridDuration(sampler_name, scripts_list)
            actions, script_string = s.get_combined_script()
            if clean_data and s.prec_ideal_transitions < info['min_ideal_transition_prec']:
                continue
            graphs, times, activities, activities_coarse, script_string, imp_obj = get_graphs(actions, script_string=script_string)
        except SamplingFailure as sf:
            if verbose:
                print (sf)
            with open(os.path.join(logs_dir,'{:03d}'.format(routine_num)+'.txt'), 'a') as f:
                f.write(str(sf)+'\n\n')
            continue
        script_file = os.path.join(scripts_dir,'{:03d}'.format(routine_num)+'.txt')


        
        # if script_use_file is not None:
        #     with open(script_use_file, 'a') as f:
        #         for t,name,act_info in s.activities:
        #             f.write('\n' + name + ';' + name+'-'+str(act_info['filename']) + ';' + str(t) + ';' + str(act_info['end_time']))
        
        with open(script_file, 'w') as f:
            try:
                f.write('Scripts used for this routine : \n'+'\n'.join(['{} : {} to {}'.format(name, act_info['time_from_h'], act_info['time_to_h']) for t,name,act_info in s.activities])+'\n\n\n')
            except:
                pass
            f.write(script_string)
        print(f'Generated script {script_file}')
        routine_out = ({'times':times, 'activities':activities, 'activities_coarse':activities_coarse, 'graphs':graphs, 'important_objects':imp_obj})
        routine_file = os.path.join(routines_dir,'{:03d}'.format(routine_num)+'.json')
        with open(routine_file, 'w') as f:
            json.dump(routine_out, f, indent=4)
        return routine_out


def main(sampler_name, output_directory, verbose, scripts_list, clean_data, sampler_specification=None):
    scripts_train_dir = os.path.join(output_directory,'scripts_train')
    scripts_test_dir = os.path.join(output_directory,'scripts_test')
    routines_raw_train_dir = os.path.join(output_directory,'raw_routines_train')
    routines_raw_test_dir = os.path.join(output_directory,'raw_routines_test')
    logs_dir_train = os.path.join(output_directory,'logs_train')
    logs_dir_test = os.path.join(output_directory,'logs_test')
    
    global info
    
    os.makedirs(output_directory)
    os.makedirs(scripts_train_dir)
    os.makedirs(scripts_test_dir)
    os.makedirs(routines_raw_train_dir)
    os.makedirs(routines_raw_test_dir)
    os.makedirs(logs_dir_train)
    os.makedirs(logs_dir_test)
    
    with open(os.path.join(output_directory,'scripts_available_to_use.txt'), 'w') as f:
        json.dump([[inf['filename'] for inf in infs] for infs in scripts_list.values()], f, indent=4)

    if sampler_name in persona_options:
        sampler = ScheduleDistributionSampler(type=sampler_name)
        sampler.plot(output_directory)
    if sampler_name == 'breakfast_or_leave':
        sampler = ScheduleSamplerBreakfastOrLeave()
        sampler.plot(output_directory)
    elif sampler_name == 'phone_interruption':
        sampler = ScheduleSamplerPhoneInterruption()
        sampler.plot(output_directory)
    
    for routine_num in range(info['num_train_routines']):
        make_routine(routine_num, scripts_train_dir, routines_raw_train_dir, sampler_name, scripts_list, logs_dir_train, output_directory, clean_data, verbose, sampler_specification=sampler_specification)
    for routine_num in range(info['num_test_routines']):
        make_routine(routine_num, scripts_test_dir, routines_raw_test_dir, sampler_name, scripts_list, logs_dir_test, output_directory, clean_data, verbose, sampler_specification=sampler_specification)

    with open(os.path.join(routines_raw_test_dir,'{:03d}'.format(0)+'.json')) as f:
        reference_full_graph = json.load(f)['graphs'][0]

    utilized_object_ids = set()
    # update utilized objects to include complete tree
    if info['only_used_objects']:
        for routine_num in range(info['num_train_routines']):
            with open(os.path.join(routines_raw_train_dir,'{:03d}'.format(routine_num)+'.json')) as f:
                utilized_object_ids.update(json.load(f)['important_objects'])
        for routine_num in range(info['num_test_routines']):
            with open(os.path.join(routines_raw_test_dir,'{:03d}'.format(routine_num)+'.json')) as f:
                utilized_object_ids.update(json.load(f)['important_objects'])
        check_ids = utilized_object_ids
        while len(check_ids) > 0:
            next_check_ids = set()
            for e in reference_full_graph['edges']:
                if e['from_id'] in check_ids and e['relation_type'] in edge_classes:
                    utilized_object_ids.add(e['to_id'])
                    next_check_ids.add(e['to_id'])
            check_ids = next_check_ids



    routines_train_dir = os.path.join(output_directory,'routines_train')
    routines_test_dir = os.path.join(output_directory,'routines_test')
    os.makedirs(routines_train_dir)
    os.makedirs(routines_test_dir)


    def postprocess(src_file, dest_file):
        with open(src_file) as f:
            datapoint = json.load(f)
        del datapoint['important_objects']
        datapoint['graphs'] = remove_ignored_classes(datapoint['graphs'], utilized_object_ids)
        with open(dest_file, 'w') as f:
            json.dump(datapoint, f, indent=4)

    for routine_num in range(info['num_train_routines']):
        postprocess(os.path.join(routines_raw_train_dir,'{:03d}'.format(routine_num)+'.json'), os.path.join(routines_train_dir,'{:03d}'.format(routine_num)+'.json'))
    for routine_num in range(info['num_test_routines']):
        postprocess(os.path.join(routines_raw_test_dir,'{:03d}'.format(routine_num)+'.json'), os.path.join(routines_test_dir,'{:03d}'.format(routine_num)+'.json'))

    with open(os.path.join(routines_test_dir,'{:03d}'.format(0)+'.json')) as f:
        refercnce_graph = json.load(f)['graphs'][0]

    nodes = refercnce_graph['nodes']
    with open(os.path.join(output_directory,'classes.json'), 'w') as f:
        global all_activities
        global all_activities_coarse
        all_activities = list(all_activities)
        all_activities.sort()
        all_activities_coarse = list(all_activities_coarse)
        all_activities_coarse.sort()
        json.dump({"nodes":nodes, "edges":edge_classes, "activities":all_activities, "activities_coarse":all_activities_coarse}, f, indent=4)

    info['num_nodes'] = len(nodes)
    search_objects = [n for n in nodes if n['id'] in utilized_object_ids and n['category']=='placable_objects']
    info['search_object_ids'] = [n['id'] for n in search_objects]
    info['search_object_names'] = [n['class_name'] for n in search_objects]
    with open(os.path.join(output_directory,'info.json'), 'w') as f:
        json.dump(info, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/try', help='Directory to output data in')
    parser.add_argument('--sampler', type=str, default='two_scenarios_simple', help='Name of schedule sampler to use. This can be \'persona\', \'individual\', \'cluster\' or an individual ID or persona name')
    parser.add_argument('--sampler_spec', type=str, default='', help='Sampler style to use. This can be \'balanced\' or \'one_at_a_time\'')
    parser.add_argument('--verbose', action='store_true', default=False, help='Set this to generate a complete dataset of all individuals and personas')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Set this to overwrite any directory!!')
    parser.add_argument('--clean_data', action='store_true', default=False)


    args = parser.parse_args()
    
    if args.clean_data:
        raise NotImplementedError('Clean data argument is not supported')

    def check_path(path):
        if os.path.exists(path):
            overwrite = input(path+' already exists. Do you want to overwrite it? (y/n)') or args.overwrite
            if overwrite.lower() == 'y':
                shutil.rmtree(path)
            else:
                raise InterruptedError()

    options_list = {
        'persona': persona_options,
        'balanced_persona': ['balanced0', 'balanced1', 'balanced2'],
        'new_persona': ['personahome', 'personawork'],
        'old_persona': ['persona0', 'persona1', 'persona2', 'persona3', 'persona4'],
        'individual': individual_options,
        'other': ['breakfast', 'scripted', 'breakfast_or_leave', 'phone_interruption', 'two_scenarios_simple']
    }

    if args.sampler in options_list.keys():
        os.makedirs(args.path)
        pool = multiprocessing.Pool()
        for n,p in enumerate(options_list[args.sampler.lower()]):
            # main(p, os.path.join(args.path,p), args.verbose, get_script_files_list(n), args.clean_data)
            out_path = os.path.join(args.path,p+'_'+args.sampler_spec)
            check_path(out_path)
            pool.apply_async(main, args=(p, out_path, args.verbose, get_scripts(n), args.clean_data, args.sampler_spec))
        pool.close()
        pool.join()
    elif args.sampler in persona_options + individual_options:
        p = args.sampler
        out_path = os.path.join(args.path,p+'_'+args.sampler_spec)
        check_path(out_path)
        main(p, out_path, args.verbose, get_scripts(0), args.clean_data, args.sampler_spec)
    elif args.sampler in options_list['other']:
        p = args.sampler
        out_path = os.path.join(args.path,p+'_'+args.sampler_spec)
        check_path(out_path)
        main(p, out_path, args.verbose, get_scripts(-1), args.clean_data, args.sampler_spec)
    else:
        raise Exception(f'{args.sampler} is not a valid sampler.')
        