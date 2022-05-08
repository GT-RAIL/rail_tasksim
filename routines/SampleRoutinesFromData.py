# ## Sample scripts for full routines from sourced scripts and sourced schedules

import json
import os
import glob
import shutil
import argparse
from math import ceil
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import sys

from requests import options
from sklearn import cluster

sys.path.append('..')
sys.path.append('../simulation')

from evolving_graph.scripts import Script
import evolving_graph.utils as utils
from evolving_graph.execution import ScriptExecutor
from evolving_graph.environment import EnvironmentGraph

from GraphReader import GraphReader, init_graph_file, scene_num
from ProgramExecutor import read_program
from ScheduleDistributionSampler import ScheduleDistributionSampler, activity_map, persona_options, individual_options
# from postprocess_viz import dump_visuals

set_seed = 23424
random.seed(set_seed)
np.random.seed(set_seed)

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

def day_num(day_of_week):
    return {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}[day_of_week]

init_graph = GraphReader(graph_file=init_graph_file)
print(f'Using scene {int(scene_num)-1}, i.e. \'TestScene{scene_num}\'')

info = {}
info['dt'] = 10   # minutes
info['num_train_routines'] = 3
info['num_test_routines'] = 1
info['weekend_days'] = []   #[day_num(day) for day in ['Saturday','Sunday']]
info['start_time'] = time_mins(mins=0, hrs=6)
info['end_time'] = time_mins(mins=0, hrs=24)
info['interleaving'] = False
info['only_used_objects'] = True
info['graphs_dt_apart'] = False


info['min_activities'] = 1
info['schedule_sampler_filter_num'] = 0 
info['idle_sampling_factor'] = 1.0
info['block_activity_for_hrs'] = 3

info['breakfast_only'] = False
info['single_script_only'] = False

ignore_classes = ['floor','wall','ceiling','character']
utilized_object_ids = set()
edge_classes = ["INSIDE", "ON"]

class SamplingFailure(Exception):
    pass

def get_scripts(n):
    scripts_list = {}
    directory = os.path.join('data/sourcedScriptsByActivity')
    for activity in os.listdir(directory):
        available_files = os.listdir(os.path.join(directory,activity))
        if not available_files:
            continue
        available_files.sort()
        nact = deepcopy(n) 
        while nact >= len(available_files):
            nact -= len(available_files)
        f = available_files[nact]
        scripts_list[activity] = {'filename': f}
    scripts_list["come_home"] = deepcopy(scripts_list["leave_home"])
    scripts_list["leaving_home_and_coming_back"] = deepcopy(scripts_list["leave_home"])
    for activity, info in scripts_list.items():
        info = read_program(os.path.join('data/sourcedScriptsByActivity',activity,info['filename']), init_graph.node_map)
        scripts_list[activity].update(info)
    return scripts_list

# %% Schedule
class ScheduleFromHistogram():
    def __init__(self, sampler_name, scripts_list, num_optional_activities=-1):
        global info
        sampler = ScheduleDistributionSampler(type=sampler_name, idle_sampling_factor=info['idle_sampling_factor'], resample_after=info['block_activity_for_hrs'], num_optional_activities=num_optional_activities)
        self.activities = []
        t = info['start_time']
        activity_name = None

        while t < info['end_time']:
            new_activity = sampler(t)
            # print(time_human(t), new_activity)
            if new_activity is None:
                if activity_name is not None:
                    self.activities[-1][2]['end_time'] = deepcopy(t)
                    activity_name = deepcopy(new_activity)
                    duration_min, duration_max = float("inf"), float("inf")
                t += info['dt']
                continue
            if activity_name == new_activity:
                t += info['dt']
                if len(self.activities) > 0 and t-self.activities[-1][0] >= duration_max:
                    sampler.update_distributions(t, activity_name)
                    self.activities[-1][2]['end_time'] = deepcopy(t)
                    activity_name = None
                    duration_min, duration_max = float("inf"), float("inf")
                continue
            if len(self.activities) > 0 and activity_name is not None:
                self.activities[-1][2]['end_time'] = deepcopy(t)
            sampler.update_distributions(t, activity_name)
            activity_name = new_activity
            self.activities.append((t, activity_name, scripts_list[activity_name]))
            duration_min, duration_max = scripts_list[activity_name]['total_duration_range']
            t += duration_min
        if activity_name is not None:
            self.activities[-1][2]['end_time'] = deepcopy(t)

        # if len(self.activities) < info['min_activities']:
        #     need = info['min_activities']
        #     raise SamplingFailure(f'Could not sample enough activities. Sampled {len(self.activities)} need at least {need}')


    def get_combined_script(self, verbose=False):
        all_actions = []
        script_header = ''
        for t, activity, info in self.activities:
            start_time = t
            end_time = info['end_time']
            # print(f'Time range for activity {start_time} to {end_time}')
            script_header += '{} ({} - {}) \n'.format(activity, time_human(start_time), time_human(end_time))
            remaining_min, remaining_max = deepcopy(info['total_duration_range'])
            duration_remaining = deepcopy(end_time - start_time)
            for act_line, act_duration in zip(info['lines'], info['durations']):
                remaining_min -= act_duration[0]
                remaining_max -= act_duration[1]
                sampling_min = max(act_duration[0], duration_remaining - remaining_max)
                sampling_max = min(act_duration[1], duration_remaining - remaining_min)
                d = (random.random() * (sampling_max-sampling_min) + sampling_min)
                duration_remaining -= d
                # print(f'Duration Remaining {duration_remaining}; Remaining {remaining_min}-{remaining_max}; Sampling {sampling_min}-{sampling_max}')
                # print('Sampled : ',t)
                all_actions.append({'script':act_line, 'time_from':t, 'time_to':t+d, 'time_from_h':time_human(t), 'time_to_h':time_human(t+d), 'name':activity+'-'+info['filename']})
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
    times = []
    important_objects = set()

    last_source = None
    for graph, action_info in zip(graph_list[1:], all_actions):
        src = action_info['name']
        if action_info['name'] != last_source:
            script_string += f'\n\n### {src}\n'
            last_source = src
        script_string += '\n' + str(action_info['script'])
        script_string += '\n## {}\n'.format(action_info['time_to_h'])
        all_rel = [edge['relation_type'] for edge in graph['edges']]
        if 'HOLDS_RH' in all_rel or 'HOLDS_LH' in all_rel:
            continue
        if len(times) > 0: # and action_info['time_to']-times[-1] <1:
            graphs[-1] = graph
        else:
            if len(graphs) > 1:
                script_string += print_graph_difference(graphs[-2], graphs[-1]) + '\n'
            graphs.append(graph)
            times.append(action_info['time_to'])
        important_objects.update(get_used_objects(graphs[-2],graphs[-1]))

    return graphs, times, script_string, list(important_objects)



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
    edges_removed = [e for e in g1['edges'] if e not in g2['edges']]
    edges_added = [e for e in g2['edges'] if e not in g1['edges']]
    nodes_removed = [n for n in g1['nodes'] if n['id'] not in [n2['id'] for n2 in g2['nodes']]]
    nodes_added = [n for n in g2['nodes'] if n['id'] not in [n2['id'] for n2 in g1['nodes']]]

    info_str = ''

    for n in nodes_removed:
        info_str += ('\n   - '+str(n))
    for n in nodes_added:
        info_str += ('\n   + '+str(n))
    for e in edges_removed:
        c1 = class_from_id(g1,e['from_id'])
        c2 = class_from_id(g1,e['to_id'])
        if c1 != 'character' and c2 != 'character' and e['relation_type'] in edge_classes:
            info_str += ('\n   - '+c1+' '+e['relation_type']+' '+c2)
    for e in edges_added:
        c1 = class_from_id(g2,e['from_id'])
        c2 = class_from_id(g2,e['to_id'])
        if c1 != 'character' and c2 != 'character' and e['relation_type'] in edge_classes:
            info_str += ('\n   + '+c1+' '+e['relation_type']+' '+c2)

    return info_str

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
def make_routine(routine_num, scripts_dir, routines_dir, sampler_name, scripts_list, logs_dir, script_use_file=None, num_optional_activities=-1, verbose=False):
    while True:
        try:
            s = ScheduleFromHistogram(sampler_name, scripts_list, num_optional_activities=num_optional_activities)
            actions, script_string = s.get_combined_script()
            graphs, times, script_string, imp_obj = get_graphs(actions, script_string=script_string)
        except SamplingFailure as sf:
            if verbose:
                print (sf)
            with open(os.path.join(logs_dir,'{:03d}'.format(routine_num)+'.txt'), 'a') as f:
                f.write(str(sf)+'\n\n')
            continue
        script_file = os.path.join(scripts_dir,'{:03d}'.format(routine_num)+'.txt')
        if script_use_file is not None:
            with open(script_use_file, 'a') as f:
                for t,name,info in s.activities:
                    f.write('\n' + name + ';' + name+'-'+str(info['filename']) + ';' + str(t) + ';' + str(info['end_time']))
        
        with open(script_file, 'w') as f:
            try:
                f.write('Scripts used for this routine : \n'+'\n'.join(['{} : {} to {}'.format(name, info['time_from_h'], info['time_to_h']) for t,name,info in s.activities])+'\n\n\n')
            except:
                pass
            f.write(script_string)
        print(f'Generated script {script_file}')
        routine_out = ({'times':times,'graphs':graphs, 'important_objects':imp_obj})
        routine_file = os.path.join(routines_dir,'{:03d}'.format(routine_num)+'.json')
        with open(routine_file, 'w') as f:
            json.dump(routine_out, f)
        return routine_out


def main(sampler_name, output_directory, verbose, scripts_list, num_opt_act):
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
        script_usage = json.dump([inf['filename'] for inf in scripts_list.values()], f, indent=4)

    sampler = ScheduleDistributionSampler(type=sampler_name, num_optional_activities=num_opt_act)
    sampler.plot(output_directory)

    # pool = multiprocessing.Pool()
    for routine_num in range(info['num_train_routines']):
        make_routine(routine_num, scripts_train_dir, routines_raw_train_dir, sampler_name, scripts_list, logs_dir_train, os.path.join(output_directory,'script_usage.txt'), num_opt_act, verbose)
        # pool.apply_async(make_routine, args = (routine_num, scripts_train_dir, routines_raw_train_dir, sampler_name, script_files_list, os.path.join(output_directory,'script_usage.txt'), verbose))
    for routine_num in range(info['num_test_routines']):
        make_routine(routine_num, scripts_test_dir, routines_raw_test_dir, sampler_name, scripts_list, logs_dir_test, os.path.join(output_directory,'script_usage.txt'), num_opt_act, verbose)
        # pool.apply_async(make_routine, args=(routine_num, scripts_test_dir, routines_raw_test_dir, sampler_name, script_files_list, os.path.join(output_directory,'script_usage.txt'), verbose))
    # pool.close()
    # pool.join()

    # use_per_script = {('_'.join(path.split('/')[-2:]))[:-4]:0 for path in glob.glob('data/sourcedScriptsByActivity/*/*.txt')}
    # activity_over_time = {act:{t:0 for t in np.arange(info['start_time'], info['end_time'], 5)} for act in activity_map.values()}
    # with open(os.path.join(output_directory,'script_usage.txt')) as f:
    #     script_usage = f.read().split('\n')
    # for script_usage_info in script_usage:
    #     if script_usage_info == '': continue
    #     script, activity, start_time, end_time = script_usage_info.split(';')
    #     use_per_script[script] += 1
    #     for t in activity_over_time[activity]:
    #         if t > float(start_time) and t < float(end_time):
    #             activity_over_time[activity][t] += 1
    
    # fig, ax = plt.subplots()
    # fig.set_size_inches(18.5, 10.5)
    # ax.bar(use_per_script.keys(), use_per_script.values())
    # _ = plt.xticks(rotation=90)
    # _ = ax.set_title('Number of times each script is used in the dataset')
    # ax.legend()
    # fig.tight_layout()
    # plt.savefig(os.path.join(output_directory, 'script_usage_histogram.jpg'))


    # num_act = len(activity_over_time)
    # fig, axs = plt.subplots(4, ceil(num_act/4))
    # fig.set_size_inches(18.5, 10.5)
    # axs = axs.reshape(-1)
    # for i,(activity, time_func) in enumerate(activity_over_time.items()):
    #     if activity is None: continue
    #     axs[i].bar(time_func.keys(), time_func.values())
    #     axs[i].set_xticks([t for t in time_func.keys() if t%180==0])
    #     axs[i].set_xticklabels([time_human(t) for t in time_func.keys() if t%180==0], rotation=90)
    #     _ = axs[i].set_title(activity)
    # fig.suptitle(sampler_name)
    # fig.tight_layout()
    # plt.savefig(os.path.join(output_directory, 'activity_time.jpg'))


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
            json.dump(datapoint, f)


    # pool = multiprocessing.Pool()
    for routine_num in range(info['num_train_routines']):
        postprocess(os.path.join(routines_raw_train_dir,'{:03d}'.format(routine_num)+'.json'), os.path.join(routines_train_dir,'{:03d}'.format(routine_num)+'.json'))
    #     pool.apply_async(postprocess, args = (os.path.join(routines_raw_train_dir,'{:03d}'.format(routine_num)+'.json'), os.path.join(routines_train_dir,'{:03d}'.format(routine_num)+'.json')))
    for routine_num in range(info['num_test_routines']):
        postprocess(os.path.join(routines_raw_test_dir,'{:03d}'.format(routine_num)+'.json'), os.path.join(routines_test_dir,'{:03d}'.format(routine_num)+'.json'))
    #     pool.apply_async(postprocess, args = (os.path.join(routines_raw_test_dir,'{:03d}'.format(routine_num)+'.json'), os.path.join(routines_test_dir,'{:03d}'.format(routine_num)+'.json')))
    # pool.close()
    # pool.join()


    with open(os.path.join(routines_test_dir,'{:03d}'.format(0)+'.json')) as f:
        refercnce_graph = json.load(f)['graphs'][0]

    nodes = refercnce_graph['nodes']
    with open(os.path.join(output_directory,'classes.json'), 'w') as f:
        json.dump({"nodes":nodes, "edges":edge_classes}, f)

    info['num_nodes'] = len(nodes)
    search_objects = [n for n in nodes if n['id'] in utilized_object_ids and n['category']=='placable_objects']
    info['search_object_ids'] = [n['id'] for n in search_objects]
    info['search_object_names'] = [n['class_name'] for n in search_objects]
    # for k,v in info.items():
    #     print(k,' : ',v)
    with open(os.path.join(output_directory,'info.json'), 'w') as f:
        json.dump(info, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/generated_routine', help='Directory to output data in')
    parser.add_argument('--sampler', type=str, default='persona0', help='Name of schedule sampler to use. This can be \'persona\', \'individual\', \'cluster\' or an individual ID or persona name')
    parser.add_argument('--loop_through_all', action='store_true', default=False, help='Set this to generate a complete dataset of all individuals and personas')
    parser.add_argument('--verbose', action='store_true', default=False, help='Set this to generate a complete dataset of all individuals and personas')
    parser.add_argument('--num_optional_activities', default=-1, type=int, help='Number of activities to do in addition to the five')

    args = parser.parse_args()

    if os.path.exists(args.path):
        overwrite = input(args.path+' already exists. Do you want to overwrite it? (y/n)')
        if overwrite.lower() == 'y':
            shutil.rmtree(args.path)
        else:
            raise InterruptedError()

    options_list = {
        'persona': persona_options,
        'individual': individual_options
    }

    if args.loop_through_all:
        os.makedirs(args.path)
        pool = multiprocessing.Pool()
        for n,p in enumerate(options_list[args.sampler.lower()]):
            # main(p, os.path.join(args.path,p), args.verbose, get_script_files_list(n), num_opt_act=args.num_optional_activities)
            pool.apply_async(main, args=(p, os.path.join(args.path,p), args.verbose, get_scripts(n), args.num_optional_activities))
        pool.close()
        pool.join()
    else:
        if args.sampler in persona_options + individual_options:
            p = args.sampler
            main(p, os.path.join(args.path,p), args.verbose, get_scripts(0), args.num_optional_activities)
        else:
            p = random.choice(options_list[args.sampler.lower()])
            main(p, os.path.join(args.path,p), args.verbose, get_scripts(0), args.num_optional_activities)
        
    # dump_visuals(args.path)