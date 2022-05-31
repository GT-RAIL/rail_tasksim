from argparse import ArgumentError
import random as random
import os
import json
from math import floor
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from postprocess_viz import color_map

with open('data/personaBasedSchedules/optimized_persona.json') as f:
    personas = json.load(f)
persona_options = list(personas.keys())

with open('data/personaBasedSchedules/corrected_histograms.json') as f:
    individual_histograms = json.load(f)
    individual_options = list(individual_histograms.keys())

with open('data/personaBasedSchedules/cluster_histograms.json') as f:
    cluster_histograms = json.load(f)

seeds = {ind:i for i,ind in enumerate(individual_options)}
so_far = len(seeds)
for i,k in enumerate(persona_options):
    seeds[k] = so_far + int(i)
seeds['custom'] = len(seeds)

activity_map = {
"brush_teeth" : "brushing_teeth",
"bathe_shower" : "showering",
"prepare_eat_breakfast" : "breakfast",
"get_dressed" : "getting_dressed",
"computer_work" : "computer_work",
"prepare_eat_lunch" : "lunch",
"leave_home" : "leave_home",
"come_home" : "come_home",
"play_music" : "playing_music",
"read" : "reading",
"take_medication" : "taking_medication",
"prepare_eat_dinner" : "dinner",
"connect_w_friends" : "socializing",
"listen_to_music" : "listening_to_music",
"clean" : "cleaning",
"clean_kitchen" : "kitchen_cleaning",
"take_out_trash" : "take_out_trash",
"do_laundry" : "laundry",
"use_restroom" : "going_to_the_bathroom",
"vacuum_clean" : "vaccuum_cleaning",
"wash_dishes" : "wash_dishes",
"watch_tv" : "watching_tv",
## unnecessary
"hand_wash_clothes" : None,
"diary_journaling" : None,
"wake_up" : None,
"sleep" : None
}
start_times = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

compulsary_activities = ["brushing_teeth","showering","breakfast","dinner","lunch"]
 
def get_opt_activities(seed):
    opt_activities = list(cluster_histograms.keys())
    opt_activities.remove("come_home")
    for act in compulsary_activities:
        opt_activities.remove(act)
    random.seed(seed)
    random.shuffle(opt_activities)
    return opt_activities

activity_lists = {k:[] for k in seeds}
for k,s in seeds.items():
    activity_lists[k] = get_opt_activities(s)

def KLdivergence(act_hist1, act_hist2):
    eps = 1e-7
    def get_dist(act_hist):
        activities = act_hist.keys()
        dist = np.array([np.array(act_hist[act]).reshape(-1) for act in activities])
        num_timeslots = dist.shape[1]
        assert dist.shape[0] == len(activities)
        assert dist.shape[1] == 18
        max = dist.sum(axis=0).max()
        dist = dist/max
        idle = np.array(1-dist.sum(axis=0)).reshape(1,-1)
        dist = np.concatenate([dist, idle], axis=0)
        dist += eps
        dist /= num_timeslots
        ind_cost = idle.sum()/num_timeslots + np.array(1-dist.sum(axis=0)).sum()/num_timeslots
        return dist, ind_cost
        
    dist_p_norm, idle_p = get_dist(act_hist1)
    dist_q_norm, idle_q = get_dist(act_hist2)
    idle_cost = (idle_p + idle_q) * 0.1
    kl_div = sum(sum(dist_p_norm * np.log(dist_p_norm/dist_q_norm)))
    return kl_div - idle_cost


class ScheduleDistributionSampler():
    def __init__(self, type, idle_sampling_factor=1.0, resample_after=float("inf"), custom_label='custom', num_optional_activities=-1):
        self.activity_histogram = {}
        with open('data/personaBasedSchedules/cluster_histograms.json') as f:
            trait_histograms = json.load(f)
        with open('data/personaBasedSchedules/corrected_histograms.json') as f:
            individual_histograms = json.load(f)
        if isinstance(type, dict):
            self.activity_histogram = type
            self.label = custom_label
        elif type in cluster_histograms.keys():
            self.label = 'Cluster'+type
            for activity, freq in cluster_histograms[type].items():
                self.activity_histogram[activity] = np.array(freq)
        elif type.upper() in individual_histograms.keys():
            self.label = type.upper()
            hist = individual_histograms[type.upper()]
            for activity, freq in hist.items():
                self.activity_histogram[activity] = np.array(freq)
        elif type.lower() in personas.keys():
            persona_name = type.lower()
            self.label = persona_name
            persona = personas[persona_name]
            for activity in persona:
                try:
                    self.activity_histogram[activity] = np.array(trait_histograms[activity][int(persona[activity])])
                except Exception as e:
                    print(activity, 'does not exist in the traits for ', persona_name, 'given trait index ',int(persona[activity]), ' available ', len(trait_histograms[activity]))
                    # raise e
        else:
            raise ArgumentError(f'Unknown value {type} for Schedule Sampler')
        
        if self.label in activity_lists:
            opt_activities = activity_lists[type]
        else:
            opt_activities = list(self.activity_histogram.keys())
            opt_activities.remove("come_home")
            for a in compulsary_activities:
                opt_activities.remove(a)
        if num_optional_activities > 0:
            if num_optional_activities < len(opt_activities):
                opt_activities = opt_activities[:num_optional_activities]
            else:
                print(f'{num_optional_activities} activities not available. Using {len(opt_activities)} available activities.')

        self.activities = compulsary_activities+opt_activities
        self.sampling_range = max(sum([np.array(self.activity_histogram[activity]) for activity in self.activities])) * idle_sampling_factor
        self.resample_after = resample_after
        self.left_house = False

    def __call__(self, t_mins):
        st_idx = start_times.index(int(floor(t_mins/60)))
        if self.left_house:
            remaining_probs = self.activity_histogram["come_home"][st_idx:]
            sample = random.random()*sum(remaining_probs)
            if remaining_probs[0] > sample:
                self.left_house = False
                return "come_home"
            else:
                return None
        sample = random.random()*self.sampling_range
        activity = None
        for act in self.activities:
            thresh = self.activity_histogram[act][st_idx]
            if thresh > sample:
                activity = act
                break
            sample -= thresh
        if activity == "leave_home":
            self.left_house = True
        return activity

    def update_distributions(self, t_mins, activity):
        try:
            st_idx = start_times.index(int(floor(t_mins/60)))
        except ValueError:
            print('Time not in range for update distribution. This should only happen once-in-a-while')
            return
        if activity is None:
            return
        end_idx = min(st_idx+self.resample_after, len(start_times))
        # print(f'Blocking {activity} from {st_idx} to {end_idx}')
        self.activity_histogram[activity][st_idx:end_idx] *= 0

    def sample_end_time(self, activity, end_time_min, end_time_max, dt):
        times = np.arange(end_time_min, end_time_max, dt)
        act_probs = [self.activity_histogram[activity][start_times.index(int(floor(t_mins/60)))] for t_mins in times]
        aggregated_probs = []
        for ap in act_probs:
            new_agg = aggregated_probs[-1] * ap if len(aggregated_probs) > 0 else ap
            aggregated_probs.append(new_agg)
        sample = random.random() * sum(aggregated_probs)
        cap = 0
        for i,ap in enumerate(aggregated_probs):
            cap += ap
            if cap > sample:
                return times[i]
        print("sampling returned ~1. This shouldn't happen very often")
        return times[-1]

    def remove(self, activity):
        self.removed_activities.append(activity)

    def plot(self, dirname = None):
        # clrs = sns.color_palette("pastel") + sns.color_palette("dark") + sns.color_palette()
        # fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        # fig3, ax3 = plt.subplots()
        # fig.set_size_inches(27, 18.5)
        fig2.set_size_inches(27, 18.5)
        # fig3.set_size_inches(27, 18.5)
        activity_list = []
        base = np.zeros_like(self.activity_histogram[self.activities[0]])
        for idx, (activity, histogram) in enumerate(self.activity_histogram.items()):
            # ax.bar(start_times, histogram, label=activity, bottom=base, color = color_map[activity])
            base += histogram
            ax2.bar(start_times, histogram, bottom=idx, color = color_map[activity])
            activity_list.append(activity)
            # ax3.plot(start_times, histogram, label=activity, color = color_map[activity], linewidth=2)
        # ax.set_xticks(start_times)
        ax2.set_xticks(start_times)
        # ax3.set_xticks(start_times)
        # ax.set_xticklabels([str(s)+':00' for s in start_times])
        ax2.set_xticklabels([str(s)+':00' for s in start_times])
        # ax3.set_xticklabels([str(s)+':00' for s in start_times])
        ax2.set_yticks(np.arange(len(activity_list)))
        ax2.set_yticklabels(activity_list)
        # ax.set_title(self.label)
        ax2.set_title(self.label)
        # ax3.set_title(self.label)
        plt.legend()
        if dirname is not None:
            # fig.savefig(os.path.join(dirname, 'sampling_distribution.jpg'))
            fig2.savefig(os.path.join(dirname, 'sampling_distribution_separated.jpg'))
            # fig3.savefig(os.path.join(dirname, 'sampling_distribution_lines.jpg'))
        else:
            plt.show()

