from argparse import ArgumentError
import random
import os
import json
from math import floor
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from postprocess_viz import color_map

personas = {}
# early riser, works long hours, has less time for chores
personas['hard_worker'] = {
    'leave_home' : 'morning',
    'come_home' : 'evening',
    'playing_music' : 'not_at_all',
    'getting_dressed' : 'morning',
    'cleaning' : 'evening',
    'breakfast' : 'has',
    'socializing' : 'not_at_all',
    'lunch' : 'skips',
    'going_to_the_bathroom' : 'uses_bathroom',
    'listening_to_music' : 'not_at_all',
    'taking_medication' : 'not_at_all',
    'take_out_trash' : 'not_at_all',
    'kitchen_cleaning' : 'not_at_all',
    'dinner' : 'has',
    'wash_dishes' : 'evening',
    'brushing_teeth' : 'multiple_times',
    'laundry' : 'not_at_all',
    'reading' : 'evening',
    'showering' : 'multiple_times',
    'computer_work' : 'less',
    'vaccuum_cleaning' : 'not_at_all',
    'watching_tv' : 'not_at_all',
}

# early riser, works from home, enjoys evenings with tv, music and friends 
personas['work_from_home'] = {
    'leave_home' : 'not_at_all',
    'come_home' : 'not_at_all',
    'playing_music' : 'evening',
    'getting_dressed' : 'evening',
    'cleaning' : 'evening',
    'breakfast' : 'has',
    'socializing' : 'evening',
    'lunch' : 'has',
    'going_to_the_bathroom' : 'uses_bathroom',
    'listening_to_music' : 'evening',
    'taking_medication' : 'multiple_times',
    'take_out_trash' : 'evening',
    'kitchen_cleaning' : 'not_at_all',
    'dinner' : 'has',
    'wash_dishes' : 'not_at_all',
    'brushing_teeth' : 'morning',
    'laundry' : 'not_at_all',
    'reading' : 'evening',
    'showering' : 'evening',
    'computer_work' : 'normal',
    'vaccuum_cleaning' : 'not_at_all',
    'watching_tv' : 'evening',
}

# home maker, less computer work, lots of chores, enjoys evenings with tv, music and friends 
personas['home_maker'] = {
    'leave_home' : 'morning',
    'come_home' : 'morning',
    'playing_music' : 'evening',
    'getting_dressed' : 'evening',
    'cleaning' : 'evening',
    'breakfast' : 'has',
    'socializing' : 'evening',
    'lunch' : 'has',
    'going_to_the_bathroom' : 'uses_bathroom',
    'listening_to_music' : 'evening',
    'taking_medication' : 'evening',
    'take_out_trash' : 'evening',
    'kitchen_cleaning' : 'morning',
    'dinner' : 'has',
    'wash_dishes' : 'morning',
    'brushing_teeth' : 'morning',
    'laundry' : 'morning',
    'reading' : 'evening',
    'showering' : 'multiple_times',
    'computer_work' : 'less',
    'vaccuum_cleaning' : 'morning',
    'watching_tv' : 'not_at_all',
}

# elderly, less work and no going out, sparsely indulges in leisurely activities 
personas['senior'] = {
    'leave_home' : 'not_at_all',
    'come_home' : 'not_at_all',
    'playing_music' : 'evening',
    'getting_dressed' : 'not_at_all',
    'cleaning' : 'not_at_all',
    'breakfast' : 'has',
    'socializing' : 'not_at_all',
    'lunch' : 'has',
    'going_to_the_bathroom' : 'uses_bathroom',
    'listening_to_music' : 'evening',
    'taking_medication' : 'multiple_times',
    'take_out_trash' : 'evening',
    'kitchen_cleaning' : 'not_at_all',
    'dinner' : 'has',
    'wash_dishes' : 'not_at_all',
    'brushing_teeth' : 'multiple_times',
    'laundry' : 'not_at_all',
    'reading' : 'evening',
    'showering' : 'multiple_times',
    'computer_work' : 'less',
    'vaccuum_cleaning' : 'morning',
    'watching_tv' : 'evening',
}

# basic 
personas['basic'] = {
    'leave_home' : 'not_at_all',
    'come_home' : 'not_at_all',
    'playing_music' : 'not_at_all',
    'getting_dressed' : 'not_at_all',
    'cleaning' : 'not_at_all',
    'breakfast' : 'skips_breakfast',
    'socializing' : 'not_at_all',
    'lunch' : 'skips_lunch',
    'listening_to_music' : 'not_at_all',
    'taking_medication' : 'not_at_all',
    'take_out_trash' : 'not_at_all',
    'kitchen_cleaning' : 'not_at_all',
    'dinner' : 'has',
    'wash_dishes' : 'not_at_all',
    'brushing_teeth' : 'morning',
    'laundry' : 'not_at_all',
    'reading' : 'not_at_all',
    'showering' : 'morning',
    'vaccuum_cleaning' : 'not_at_all',
    'watching_tv' : 'not_at_all',
}


persona_options = list(personas.keys())
with open('data/personaBasedSchedules/individual_histograms.json') as f:
    individual_histograms = json.load(f)
    individual_options = list(individual_histograms.keys())

seeds = {ind:i for i,ind in enumerate(individual_options)}
seeds['hard_worker'] = len(individual_options)
seeds['home_maker'] = len(individual_options) + 1
seeds['work_from_home'] = len(individual_options) + 2
seeds['senior'] = len(individual_options) + 3


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

def KLdivergence(act_hist1, act_hist2):
    eps = 1e-7
    activities = act_hist1.keys()
    dist_p = np.array([act_hist1[act] for act in activities]) + eps
    dist_p_norm = dist_p/sum(sum(dist_p))
    # assert(np.isclose(sum(sum(dist_p)), len(start_times))), str(sum(sum(dist_p))) + ' , ' + str(len(start_times))
    dist_q = np.array([act_hist2[act] for act in activities]) + eps
    kl_div = sum(sum(dist_p_norm * np.log(dist_p/dist_q)))
    return kl_div

class ScheduleDistributionSampler():
    def __init__(self, type, idle_sampling_factor=1.0, resample_after=float("inf")):
        self.activity_histogram = {}
        with open('data/personaBasedSchedules/trait_histograms.json') as f:
            trait_histograms = json.load(f)
        with open('data/personaBasedSchedules/individual_histograms.json') as f:
            individual_histograms = json.load(f)
        if type.upper() in individual_histograms.keys():
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
                    self.activity_histogram[activity] = np.array(trait_histograms[activity][persona[activity]])
                except Exception as e:
                    print(activity, ' does not exist in the traits for ', persona_name)
                    # raise e
        else:
            raise ArgumentError(f'Unknown value {type} for Schedule Sampler')
        
        self.activities = list(self.activity_histogram.keys())
        self.activities.remove("come_home")
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
        self.update_distributions(st_idx, activity)
        if activity == "leave_home":
            self.left_house = True
        return activity

    def update_distributions(self, st_idx, activity):
        if activity is None:
            return
        end_idx = min(st_idx+self.resample_after, len(start_times))
        self.activity_histogram[activity][st_idx:end_idx] *= 0

    def remove(self, activity):
        self.removed_activities.append(activity)

    def plot(self, dirname = None):
        # clrs = sns.color_palette("pastel") + sns.color_palette("dark") + sns.color_palette()
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig.set_size_inches(27, 18.5)
        fig2.set_size_inches(27, 18.5)
        fig3.set_size_inches(27, 18.5)
        activity_list = []
        base = np.zeros_like(self.activity_histogram[self.activities[0]])
        for idx, (activity, histogram) in enumerate(self.activity_histogram.items()):
            ax.bar(start_times, histogram, label=activity, bottom=base, color = color_map[activity])
            base += histogram
            ax2.bar(start_times, histogram, bottom=idx, color = color_map[activity])
            activity_list.append(activity)
            ax3.plot(start_times, histogram, label=activity, color = color_map[activity], linewidth=2)
        ax.set_xticks(start_times)
        ax2.set_xticks(start_times)
        ax3.set_xticks(start_times)
        ax.set_xticklabels([str(s)+':00' for s in start_times])
        ax2.set_xticklabels([str(s)+':00' for s in start_times])
        ax3.set_xticklabels([str(s)+':00' for s in start_times])
        ax2.set_yticks(np.arange(len(activity_list)))
        ax2.set_yticklabels(activity_list)
        ax.set_title(self.label)
        ax2.set_title(self.label)
        ax3.set_title(self.label)
        plt.legend()
        if dirname is not None:
            fig.savefig(os.path.join(dirname, 'sampling_distribution.jpeg'))
            fig2.savefig(os.path.join(dirname, 'sampling_distribution_separated.jpeg'))
            fig3.savefig(os.path.join(dirname, 'sampling_distribution_lines.jpeg'))
        else:
            plt.show()

