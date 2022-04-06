# ## Sample scripts for full routines from sourced scripts and sourced schedules

import json
import os
import glob
import shutil
import argparse
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import sys

from sklearn.utils import resample
sys.path.append('..')
sys.path.append('../simulation')

from evolving_graph.scripts import Script
import evolving_graph.utils as utils
from evolving_graph.execution import ScriptExecutor
from evolving_graph.environment import EnvironmentGraph

from GraphReader import GraphReader, init_graph_file, scene_num
from ProgramExecutor import read_program
from ScheduleDistributionSampler import ScheduleDistributionSampler, activity_map, persona_options, individual_options
from postprocess_viz import dump_visuals

# random.seed(23424)

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
info['num_train_routines'] = 50
info['num_test_routines'] = 10
info['weekend_days'] = []   #[day_num(day) for day in ['Saturday','Sunday']]
info['start_time'] = time_mins(mins=0, hrs=6)
info['end_time'] = time_mins(mins=0, hrs=24)
info['interleaving'] = False
info['only_used_objects'] = True
info['graphs_dt_apart'] = False


info['min_activities'] = 1
info['schedule_sampler_filter_num'] = 0 
info['idle_sampling_factor'] = 1.0
info['block_activity_for_hrs'] = 5

info['breakfast_only'] = False
info['single_script_only'] = False

ignore_classes = ['floor','wall','ceiling','character']
utilized_object_ids = set()
edge_classes = ["INSIDE", "ON"]


class SamplingFailure(Exception):
    pass

def get_script_files_list():
    files_list = {}
    directory = os.path.join('data/sourcedScriptsByActivity')
    for activity in os.listdir(directory):
        available_files = os.listdir(os.path.join(directory,activity))
        if not available_files:
            continue
        f = random.choice(available_files)
        files_list[activity] = f
    files_list["come_home"] = files_list["leave_home"]
    files_list["leaving_home_and_coming_back"] = files_list["leave_home"]
    return files_list

# %% Activity

class Activity():
    def __init__(self, name, time_start_mins, time_end_mins=None, script_file=None, verbose=False, stack_actions=True):
        self.name = name
        directory = os.path.join('data/sourcedScriptsByActivity', name)
        if script_file is None:
            try:
                if info['single_script_only']:
                    script_files = os.listdir(directory)
                    script_files.sort()
                    script_file = script_files[0]
                else:
                    script_file = np.random.choice(os.listdir(directory))
            except:
                raise SamplingFailure(f'No script found for {name}')
            if verbose:
                print(f'Picking script {script_file} for {name}')
            # headers, self.scripts, self.obj_use, _ = read_program(os.path.join(directory,script_file), init_graph.node_map)
        durations, self.scripts, self.obj_start, self.obj_end = read_program(os.path.join(directory,script_file), init_graph.node_map)
        self.source = '_'.join([name, script_file[:-4]])

        self.durations = [(random.random() * (duration_max-duration_min) + duration_min) for duration_max,duration_min in durations]
        def valid_times(times):
            for time, next_time, duration in zip(times[:-1], times[1:], self.durations[:-1]):
                if next_time - time <= duration:
                    return False
                if info['graphs_dt_apart'] and next_time - time <= info['dt']:
                    return False
            return True
        if stack_actions:
            times = [time_start_mins]
            if time_end_mins is not None:
                if time_start_mins >= time_end_mins - sum(self.durations):
                    raise SamplingFailure(f'Cannot sample {sum(self.durations)}mins long activity {self.source} within {time_start_mins} and {time_end_mins}')
                times[0] = random.randrange(time_start_mins, time_end_mins - sum(self.durations))
            for d in self.durations:
                times.append(times[-1]+d)
        else:
            assert time_end_mins is not None, 'You need to either stack actions or provide an end time.'
            for _ in range(5):
                times = (np.random.rand(len(self.lines)) * (time_end_mins-time_start_mins) + time_start_mins).round().astype(int)
                times.sort()
                if valid_times(times):
                    break
            if not valid_times(times):
                raise SamplingFailure(f'Invalid times {times} and durations {self.durations} for activity {self.source}')
        self.times = times
        self.source += '('+time_human(self.times[0])+' - '+time_human(self.get_end_time())+') '
    
    def get_action_info(self):
        detailed_actions = []
        for t,d,scr,obj_s,obj_e in zip(self.times, self.durations, self.scripts, self.obj_start, self.obj_end):
            t2 = t+d
            detailed_actions.append({'time_from':t, 'time_to':t2, 'name':self.source, 'script':scr, 'time_from_h':time_human(t), 'time_to_h':time_human(t2), 'start_using':obj_s, 'end_using':obj_e, 'source':self.source})
        return detailed_actions
    
    def get_end_time(self):
        return self.times[-1] + self.durations[-1]

    def get_start_time(self):
        return self.times[0]


# %% Schedule


class Schedule():
    def __init__(self, type=None):
        pass

    def get_combined_script(self, verbose=False):
        all_actions = []
        for act in self.activities:
            all_actions += act.get_action_info()
        all_actions.sort(key = lambda x : x['time_from'])
        if verbose:
            for a in all_actions:
                print (a['time_from_h']+' to '+a['time_to_h']+' : '+a['name'])
                print ('Started using : '+str(a['start_using'])+'; Finished using : '+str(a['end_using']))
                for l in a['script']:
                    print(' - ',l)
        for end_time, next_start in zip([a['time_to'] for a in all_actions[:-1]], [a['time_from'] for a in all_actions[1:]]):
            if end_time > next_start:
                raise SamplingFailure(f'Timing conflict : End time {end_time} of an action , exceeds start time {next_start} of next action')
        return all_actions

# class ScheduleFromFile(Schedule):
#     def __init__(self, type=None):
#         assert type == None or type in ['weekday','weekend']
#         schedule_options = []
#         if info['breakfast_only']: 
#             self.schedule_file_path = 'data/sourcedSchedules/breakfast/dummy.json' 
#         else:
#             for (root,_,files) in os.walk('data/sourcedSchedules'):
#                 if type is not None and type not in root:
#                     continue
#                 schedule_options += [os.path.join(root,f) for f in files]
#             self.schedule_file_path = np.random.choice(schedule_options)
#         with open(self.schedule_file_path) as f:
#             schedule = json.load(f)
        
#         def sample_act_time(act_time_options):
#             tr = act_time_options[random.randrange(len(act_time_options))]
#             return [[tr[0],0],[tr[1],0]]
        
#         activity_with_sampled_time = [(act_name, sample_act_time(act_time_options)) for act_name,act_time_options in schedule.items()]
#         time_start_mins = time_mins(act_time[0][1], act_time[0][0])
#         time_end_mins = time_mins(act_time[1][1], act_time[1][0])
#         self.activities = [Activity(act_name, time_start_mins=time_start_mins, time_end_mins=time_end_mins, stack_actions = not info['interleaving']) for act_name,act_time in activity_with_sampled_time]
#         random.shuffle(self.activities)
#         num_activities = min(info['max_activities_per_day'], len(self.activities))
#         self.activities = self.activities[:num_activities]
    
class ScheduleFromHistogram(Schedule):
    def __init__(self, sampler_name, script_files_list, type=None):
        sampler = ScheduleDistributionSampler(type=sampler_name, idle_sampling_factor=info['idle_sampling_factor'], resample_after=info['block_activity_for_hrs'])
        if info['breakfast_only']:
            t = sampler.sample_time_for('breakfast')
            self.activities = [Activity('breakfast', time_start_mins=t, stack_actions=True, script_file=script_files_list['breakfast'])]
        else:
            # first_activity = "getting_out_of_bed"
            # first_activity_done = False
            # last_activity = "sleeping"
            self.activities = []
            t = info['start_time']
            while t < info['end_time']:
                activity_name = sampler(t)
                if activity_name is None:
                    t += info['dt']
                    continue
                # if not first_activity_done:
                #     if activity_name != first_activity:
                #         t += info['dt']
                #         continue
                #     else:
                #         first_activity_done = True
                new_activity = Activity(activity_name, time_start_mins=t, stack_actions=True, script_file=script_files_list[activity_name])
                self.activities.append(new_activity)
                t = new_activity.get_end_time() + 1
                # if activity_name == last_activity:
                #     return
            # if not first_activity_done:
            #     raise SamplingFailure('Histogram sampler could not sample {first_activity}')
            # raise SamplingFailure('Histogram sampler could not sample {last_activity}')
            if len(self.activities) < info['min_activities']:
                need = info['min_activities']
                raise SamplingFailure(f'Could not sample enough activities. Sampled {len(self.activities)} need at least {need}')

# %% run and get graphs


def get_graphs(all_actions, verbose=False):
    with open (init_graph_file,'r') as f:
        init_graph_dict = json.load(f)
    name_equivalence = utils.load_name_equivalence()
    complete_objects_in_use = [[]]
    current_objects = []
    complete_script_lines = []
    complete_script_timestamps = []
    complete_sources = []
    important_objects = set()
    for action in all_actions:
        if verbose:
            print('## Executing '+action['name']+' from '+action['time_from_h']+' to '+action['time_to_h'])
            print ('Started using : '+str(action['start_using'])+'; Finished using : '+str(action['end_using']))
            print (action['script'])
        ## add to the script
        complete_script_lines.append(action['script'])
        complete_script_timestamps.append(action['time_to'])
        complete_sources.append(action['source'])
        ## update the list of objects currently in use
        for obj in action['start_using']:
            current_objects.append(obj)
        try:
            for obj in action['end_using']:
                current_objects.remove(obj)
        except ValueError as e:
            raise ValueError('Failed to execute '+str(action['name'])+' ill-defined usage tags for '+str(obj))
        ## save the iteration results
        complete_objects_in_use.append(current_objects)

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
    obj_in_use = []
    script_string = ''

    combined_obj_in_use = set()
    last_source = None
    for graph, time, objects, line, src in zip(graph_list[1:], complete_script_timestamps, complete_objects_in_use, complete_script_lines, complete_sources):
        combined_obj_in_use.update(objects)
        if src != last_source:
            script_string += f'\n\n### {src}\n'
            last_source = src
        script_string += f'\n{line}'
        all_rel = [edge['relation_type'] for edge in graph['edges']]
        if 'HOLDS_RH' in all_rel or 'HOLDS_LH' in all_rel:
            continue
        if len(times) > 0 and time-times[-1] <1:
            graphs[-1] = graph
            # updated_obj_in_use = set(obj_in_use[-1])
            # updated_obj_in_use.update(combined_obj_in_use)
            # obj_in_use[-1] = list(updated_obj_in_use)
        else:
            if len(obj_in_use) > 0:
                script_string += '\n## Was using objects : {}'.format(str(obj_in_use[-1]))
            script_string += f'\n## {time_human(time)}\n'
            graphs.append(graph)
            times.append(time)
            obj_in_use.append([])
            # obj_in_use.append(list(combined_obj_in_use))
            # combined_obj_in_use = set()
        important_objects.update(get_used_objects(graphs[-2],graphs[-1]))
        script_string += print_graph_difference(graphs[-2], graphs[-1]) + '\n'
        if verbose:
            print('Currently using : ',current_objects)
            input('Press something...')

    return graphs, times, obj_in_use, script_string, list(important_objects)



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
        info_str += ('\nRemoved node : '+str(n))
    for n in nodes_added:
        info_str += ('\nAdded node   : '+str(n))
    for e in edges_removed:
        c1 = class_from_id(g1,e['from_id'])
        c2 = class_from_id(g1,e['to_id'])
        if c1 != 'character' and c2 != 'character' and e['relation_type'] in edge_classes:
            info_str += ('\nRemoved : '+c1+' '+e['relation_type']+' '+c2)
    for e in edges_added:
        c1 = class_from_id(g2,e['from_id'])
        c2 = class_from_id(g2,e['to_id'])
        if c1 != 'character' and c2 != 'character' and e['relation_type'] in edge_classes:
            info_str += ('\nAdded   : '+c1+' '+e['relation_type']+' '+c2)

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
def make_routine(routine_num, scripts_dir, routines_dir, sampler_name, script_files_list, logs_dir, script_use_file=None, verbose=False):
    while True:
        # day = np.random.choice(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        # day_number = day_num(day)
        # day_type = 'weekend' if day_number in info['weekend_days'] else 'weekday'
        try:
            s = ScheduleFromHistogram(sampler_name, script_files_list)
            actions = s.get_combined_script()
            graphs, times, obj_in_use, script_string, imp_obj = get_graphs(actions)
        except SamplingFailure as sf:
            if verbose:
                print (sf)
            with open(os.path.join(logs_dir,'{:03d}'.format(routine_num)+'.txt'), 'a') as f:
                f.write(str(sf)+'\n\n')
            continue
        script_file = os.path.join(scripts_dir,'{:03d}'.format(routine_num)+'.txt')
        if script_use_file is not None:
            with open(script_use_file, 'a') as f:
                for a in s.activities:
                    f.write('\n' + a.source[:a.source.index('(')] + ';' + a.name + ';' + str(a.get_start_time()) + ';' + str(a.get_end_time()))
        
        with open(script_file, 'w') as f:
            # try:
            #     f.write(day+' schedule generated from '+s.schedule_file_path)
            # except:
            #     pass
            try:
                f.write('Scripts used for this routine : \n'+'\n'.join([a.source for a in s.activities])+'\n\n\n')
            except:
                pass
            f.write(script_string)
        print(f'Generated script {script_file}')
        routine_out = ({'times':times,'graphs':graphs, 'objects_in_use':obj_in_use, 'important_objects':imp_obj})
        routine_file = os.path.join(routines_dir,'{:03d}'.format(routine_num)+'.json')
        with open(routine_file, 'w') as f:
            json.dump(routine_out, f)
        return routine_out


def main(sampler_name, output_directory, verbose, script_files_list):
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
        script_usage = json.dump(script_files_list, f)

    sampler = ScheduleDistributionSampler(type=sampler_name)
    sampler.plot(os.path.join(output_directory,'schedule_distribution.jpg'))

    pool = multiprocessing.Pool()
    for routine_num in range(info['num_train_routines']):
        make_routine(routine_num, scripts_train_dir, routines_raw_train_dir, sampler_name, script_files_list, logs_dir_train, os.path.join(output_directory,'script_usage.txt'), verbose)
        # pool.apply_async(make_routine, args = (routine_num, scripts_train_dir, routines_raw_train_dir, sampler_name, script_files_list, os.path.join(output_directory,'script_usage.txt'), verbose))
    for routine_num in range(info['num_test_routines']):
        make_routine(routine_num, scripts_test_dir, routines_raw_test_dir, sampler_name, script_files_list, logs_dir_test, os.path.join(output_directory,'script_usage.txt'), verbose)
        # pool.apply_async(make_routine, args=(routine_num, scripts_test_dir, routines_raw_test_dir, sampler_name, script_files_list, os.path.join(output_directory,'script_usage.txt'), verbose))
    pool.close()
    pool.join()

    use_per_script = {('_'.join(path.split('/')[-2:]))[:-4]:0 for path in glob.glob('data/sourcedScriptsByActivity/*/*.txt')}
    activity_over_time = {act:{t:0 for t in np.arange(info['start_time'], info['end_time'], 5)} for act in activity_map.values()}
    with open(os.path.join(output_directory,'script_usage.txt')) as f:
        script_usage = f.read().split('\n')
    for script_usage_info in script_usage:
        if script_usage_info == '': continue
        script, activity, start_time, end_time = script_usage_info.split(';')
        use_per_script[script] += 1
        for t in activity_over_time[activity]:
            if t > float(start_time) and t < float(end_time):
                activity_over_time[activity][t] += 1
    
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.bar(use_per_script.keys(), use_per_script.values())
    _ = plt.xticks(rotation=90)
    _ = ax.set_title('Number of times each script is used in the dataset')
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_directory, 'script_usage_histogram.jpg'))


    num_act = len(activity_over_time)
    fig, axs = plt.subplots(4, ceil(num_act/4))
    fig.set_size_inches(18.5, 10.5)
    axs = axs.reshape(-1)
    for i,(activity, time_func) in enumerate(activity_over_time.items()):
        if activity is None: continue
        axs[i].bar(time_func.keys(), time_func.values())
        axs[i].set_xticks([t for t in time_func.keys() if t%180==0])
        axs[i].set_xticklabels([time_human(t) for t in time_func.keys() if t%180==0], rotation=90)
        _ = axs[i].set_title(activity)
    fig.suptitle(sampler_name)
    fig.tight_layout()
    plt.savefig(os.path.join(output_directory, 'activity_time.jpg'))


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
    for k,v in info.items():
        print(k,' : ',v)
    with open(os.path.join(output_directory,'info.json'), 'w') as f:
        json.dump(info, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/generated_routine', help='Directory to output data in')
    parser.add_argument('--sampler', type=str, help='Name of schedule sampler to use. This can be \'persona\', \'individual\' or an individual ID or persona name')
    parser.add_argument('--loop_through_all', action='store_true', default=False, help='Set this to generate a complete dataset of all individuals and personas')
    parser.add_argument('--verbose', action='store_true', default=False, help='Set this to generate a complete dataset of all individuals and personas')

    args = parser.parse_args()

    if os.path.exists(args.path):
        overwrite = input(args.path+' already exists. Do you want to overwrite it? (y/n)')
        if overwrite:
            shutil.rmtree(args.path)
        else:
            raise InterruptedError()

    script_files = get_script_files_list()

    if args.loop_through_all:
        os.makedirs(args.path)
        if args.sampler.lower() == 'persona':
            for p in persona_options:
                main(p, os.path.join(args.path,p), args.verbose, script_files)
        if args.sampler.lower() == 'individual':
            for i in individual_options:
                main(i, os.path.join(args.path,i), args.verbose, script_files)
    else:
        if args.sampler.lower() == 'persona':
            main(random.choice(persona_options), args.path, args.verbose, script_files)
        elif args.sampler.lower() == 'individual':
            main(random.choice(individual_options), args.path, args.verbose, script_files)
        else:
            main(random.choice(persona_options + individual_options), args.path, args.verbose, script_files)
    
    dump_visuals(args.path)