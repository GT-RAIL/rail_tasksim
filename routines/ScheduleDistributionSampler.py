from argparse import ArgumentError
import random as random
import os
import json
from math import floor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('data/personaBasedSchedules/optimized_persona.json') as f:
    personas = json.load(f)
persona_options = list(personas.keys())

with open('data/personaBasedSchedules/corrected_histograms.json') as f:
    individual_histograms = json.load(f)
    individual_options = list(individual_histograms.keys())

with open('data/personaBasedSchedules/cluster_histograms.json') as f:
    cluster_histograms = json.load(f)

# seeds = {ind:i for i,ind in enumerate(individual_options)}
# so_far = len(seeds)
# for i,k in enumerate(persona_options):
#     seeds[k] = so_far + int(i)
# seeds['custom'] = len(seeds)

grey = sns.color_palette(palette='pastel')[7]
# %%
color_map = {
 "sleeping" : sns.color_palette(palette='pastel')[0], 
 "sleep" : sns.color_palette(palette='pastel')[0], 
 "nap" : sns.color_palette(palette='pastel')[0], 
 "wake_up" : sns.color_palette()[0], 
 "breakfast" : sns.color_palette()[1], 
 "lunch" : sns.color_palette()[2], 
 "computer_work" : sns.color_palette()[3], 
 "reading" : sns.color_palette()[4],
 "cleaning" : sns.color_palette()[5], 
 "laundry" : sns.color_palette()[6], 
 "leave_home" : sns.color_palette()[7], 
 "come_home" : sns.color_palette()[8], 
 "socializing" : sns.color_palette(palette='dark')[0], 
 "taking_medication" : sns.color_palette(palette='dark')[1], 
 "vaccuum_cleaning" : sns.color_palette(palette='dark')[2], 
 "getting_dressed" : sns.color_palette(palette='dark')[3],
 "dinner" : sns.color_palette(palette='dark')[4], 
 "kitchen_cleaning" : sns.color_palette(palette='dark')[5],
 "take_out_trash" : sns.color_palette(palette='dark')[6],
 "wash_dishes" : sns.color_palette(palette='dark')[7],
 "wash_dishes_breakfast" : sns.color_palette(palette='dark')[7],
 "wash_dishes_lunch" : sns.color_palette(palette='dark')[7],
 "wash_dishes_dinner" : sns.color_palette(palette='dark')[7],
 "playing_music" : sns.color_palette(palette='dark')[8],
 "diary_logging" : sns.color_palette(palette='dark')[9],

#  "brushing_teeth" : grey, 
#  "showering" : grey, 
#  "leaving_home_fast" : grey, 
#  "watching_tv" : grey, 
#  "talk_on_phone" : grey, 
#  "online_meeting" : grey, 
#  "going_to_the_bathroom" : grey,
#  "listening_to_music" : grey,

 "brushing_teeth" : sns.color_palette(palette='pastel')[1], 
 "showering" : sns.color_palette(palette='pastel')[2], 
 "leaving_home_fast" : sns.color_palette(palette='pastel')[3], 
 "watching_tv" : sns.color_palette(palette='pastel')[4],
 "talk_on_phone" : sns.color_palette(palette='pastel')[5], 
 "online_meeting" : sns.color_palette(palette='pastel')[6], 
 "going_to_the_bathroom" : sns.color_palette(palette='pastel')[7],
 "listening_to_music" : sns.color_palette(palette='pastel')[8],


 "breakfast_food" : sns.color_palette()[0], 
 "breakfast_food-Cereal.txt" : sns.color_palette()[1], 
 "breakfast_food-Egg.txt" : sns.color_palette()[2], 
 "breakfast_food-BreadButter.txt" : sns.color_palette()[3], 
 "breakfast_beverage" : sns.color_palette()[4], 
 "breakfast_beverage-Tea.txt" : sns.color_palette()[5], 
 "breakfast_beverage-Coffee.txt" : sns.color_palette()[6], 
 "leaving_home_fast-00.txt" : sns.color_palette()[7], 
 "talk_on_phone-00.txt" : sns.color_palette()[8], 
}

activity_minisequences = {
    "wake_up" : [
        "sleep",
        "nap",
        "computer_work",
        "reading",
        "cleaning",
        "socializing",
        "taking_medication",
        "vaccuum_cleaning",
        "take_out_trash",
        "playing_music",
        # "diary_logging",
        "brushing_teeth",
        "watching_tv",
        "going_to_the_bathroom",
        "listening_to_music",
        ],
    
    "brushing_teeth" : [
        "breakfast",
        "showering",
        "dinner",
        "lunch",
        ],
    
    "showering" : [
        "getting_dressed",
        ],

    "getting_dressed" : [
        "leave_home",
        "come_home",
        "getting_dressed"
        ],

    "come_home" : [
        "getting_dressed",
        ],

    "breakfast" : [
        "wash_dishes_breakfast",
        "kitchen_cleaning",
        ],

    "lunch" : [
        "wash_dishes_lunch",
        "kitchen_cleaning",
        ],

    "dinner" : [
        "wash_dishes_dinner",
        "kitchen_cleaning",
        "brushing_teeth",
        ],
    
    "wash_dishes_breakfast" : [
        ],
    "wash_dishes_lunch" : [
        ],
    "wash_dishes_dinner" : [
        ],
    "taking_medication" : [
        ],
    "vaccuum_cleaning" : [
        ],
    "cleaning" : [
        ],
    "socializing" : [
        ],
    "take_out_trash" : [
        ],
    "laundry" : [
        ],
    "diary_logging" : [
        ],
    
    }

activity_type = {
 "everyday": [
    "sleeping", 
    "wake_up", 
    "breakfast", 
    "lunch", 
    "taking_medication", 
    "dinner", 
    "brushing_teeth", 
    "showering", 
    "going_to_the_bathroom",
 ],
 "chores": [
    "cleaning", 
    "laundry", 
    "vaccuum_cleaning", 
    "kitchen_cleaning",
    "take_out_trash",
    "wash_dishes",
 ],
 "work_and_errands": [
    "computer_work", 
    "leave_home", 
    # "come_home", 
    "getting_dressed",
    "leaving_home_fast", 
 ],
 "pastimes": [
    "reading",
    "socializing", 
    "playing_music",
    "watching_tv", 
    "listening_to_music",
    "nap",
    "diary_logging"
 ],
 "interruption": [
    "talk_on_phone", 
    "online_meeting", 
 ]
}


start_times = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

compulsary_activities = ["brushing_teeth","showering","breakfast","dinner","lunch"]
interruption_activities = ["talk_on_phone", "going_to_the_bathroom", "online_meeting", "listen_to_music"]
 
# def get_opt_activities(seed):
#     opt_activities = list(cluster_histograms.keys())
#     opt_activities.remove("come_home")
#     for act in compulsary_activities:
#         opt_activities.remove(act)
#     random.seed(seed)
#     random.shuffle(opt_activities)
#     return opt_activities

# activity_lists = {k:[] for k in seeds}
# for k,s in seeds.items():
#     activity_lists[k] = get_opt_activities(s)

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

class ScheduleSampler_FixedSequence():
    def __init__(self) -> None:
        self.activity_idx = -1
        self.activity_seq = ["brushing_teeth", "breakfast", None, "reading", None, "lunch", "computer_work", None, "dinner", None]
        self.left_house = False
        self.update_distributions = lambda x, y: None

    def reset(self):
        self.activity_idx = -1

    def __call__(self, t) -> str:
        if self.activity_idx < len(self.activity_seq) - 1:
            self.activity_idx += 1
        return self.activity_seq[self.activity_idx]

class ScheduleDistributionSampler():
    def __init__(self, type, idle_sampling_factor=1.0, resample_after=float("inf"), custom_label='custom', num_optional_activities=-1, specification=''):
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
                # try:
                self.activity_histogram[activity] = np.array(trait_histograms[activity][int(persona[activity])])
                # except Exception as e:
                #     print(activity, 'does not exist in the traits for ', persona_name, 'given trait index ',int(persona[activity]), ' available ', len(trait_histograms[activity]))
                    # raise e
        else:
            raise Exception(f'Unknown value {type} for Schedule Sampler')
        
        self.activity_histogram['wash_dishes_breakfast'] = list(self.activity_histogram['wash_dishes'] + self.activity_histogram['breakfast'])
        self.activity_histogram['wash_dishes_lunch'] = list(self.activity_histogram['wash_dishes'] + self.activity_histogram['lunch'])
        self.activity_histogram['wash_dishes_dinner'] = list(self.activity_histogram['wash_dishes'] + self.activity_histogram['dinner'])
        self.activity_histogram['watching_tv'] /= 3
        self.activity_histogram['computer_work'] /= 3
        # if self.label in activity_lists:
        #     opt_activities = activity_lists[type]
        # else:
        #     opt_activities = list(self.activity_histogram.keys())
        #     opt_activities.remove("come_home")
        #     for a in compulsary_activities:
        #         opt_activities.remove(a)
        # if num_optional_activities > 0:
        #     if num_optional_activities < len(opt_activities):
        #         opt_activities = opt_activities[:num_optional_activities]
        #     else:
        #         print(f'{num_optional_activities} activities not available. Using {len(opt_activities)} available activities.')

        self.activities = ["wake_up"]
        # self.interruption_activities = [a for a in self.activities if a in activity_type["interruption"]]
        # if specification.lower() == 'one_at_a_time':
        #     today = random.choice(["chores", "work_and_errands", "pastimes"])
        #     print(today, end=' ')
        #     self.activities = [a for a in self.activities if a in activity_type["everyday"]+activity_type["interruption"]+activity_type[today]]
        # elif specification.lower() == 'balanced':
        #     chosen_activities = []
        #     max_num_act_per_type = 2
        #     for optional_types in ["chores", "work_and_errands", "pastimes"]:
        #         chosen_activities_of_type = [a for a in self.activities if a in activity_type[optional_types]]
        #         if len(chosen_activities_of_type) > max_num_act_per_type:
        #             random.shuffle(chosen_activities_of_type)
        #             chosen_activities_of_type = chosen_activities_of_type[:max_num_act_per_type]
        #         chosen_activities += chosen_activities_of_type
        #     self.activities = [a for a in self.activities if a in [activity_type["everyday"]+activity_type["interruption"]]] + chosen_activities
        self.sampling_range = max(sum([np.array(self.activity_histogram[act]) for act in self.activities]))
        self.resample_after = resample_after
        self.left_house = False

    def __call__(self, t_mins, interruption=False):
        st_idx = start_times.index(min(23,int(floor(t_mins/60))))
        if self.left_house:
            # assert not interruption
            remaining_probs = self.activity_histogram["come_home"][st_idx:]
            sample = random.random()*sum(remaining_probs)
            if remaining_probs[0] > sample:
                self.left_house = False
                return "come_home"
            else:
                return None
        sample = random.random()*self.sampling_range
        activity = None
        activities_and_weights = [(act,self.activity_histogram[act][st_idx]) for act in self.activities]
        # sample *= sum([w for (a,w) in activities_and_weights])
        for act, weight in activities_and_weights:
            if weight > sample:
                activity = act
                break
            sample -= weight
        if activity in activity_minisequences:
            self.activities.remove(activity)
            self.activities += activity_minisequences[activity]
        if activity == "leave_home":
            self.left_house = True
        return activity

    def valid_end(self, t, failure=None):
        if self.left_house and t >= 23*60:
            failure('Out too late!')
            return False            
        return True

    def get_probability(self, activity, t_mins):
        st_idx = start_times.index(int(floor(t_mins/60)))
        return self.activity_histogram[activity][st_idx]

    def update_distributions(self, t_mins, activity):
        if activity == 'sleeping': return
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

    def does(self, activity):
        return activity in self.activity_histogram and sum(self.activity_histogram[activity]) > 0

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


# %%
