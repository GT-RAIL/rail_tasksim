from argparse import ArgumentError
import random as random
import os
import json
from math import floor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from postprocess_viz import color_map

compulsary_activities = ["brushing_teeth","showering","breakfast","dinner","lunch"]

class ScheduleSamplerNoisy():
    def __init__(self):
        self.activities = {'social':{'options':['socializing'], 'prob':lambda t:min(min(abs(t-18*60), abs(t-21*60)), abs(t-24*60))/60*(2/3)}, 
                           'meal': {'options':['dinner'], 'prob':lambda t:min(abs(t-18*60), abs(t-24*60))/60*(1/3)}, 
                           'hobby':{'options':['watching_tv','playing_music'], 'prob':lambda t:1/6},
                           }
        self.max_prob = 1.5
        self.left_house = False

    def __call__(self, t_mins):
        sample = random.random() * self.max_prob
        activity_category = None
        for act in self.activities:
            thresh = self.get_probability(act, t_mins)
            if thresh > sample:
                activity_category = act
                break
            sample -= thresh
        activity = random.choice(self.activities[activity_category]['options']) if activity_category is not None else None
        return activity

    def update_distributions(self, t_mins, activity):
        pass

    def get_probability(self, activity, t_mins):
        return self.activities[activity]['prob'](t_mins)

class ScheduleSamplerThreeActivities():
    def __init__(self):
        if random.random() > 0.5:
            self.activities = {
                           'brushing_teeth': lambda t:(t/60)>8 and (t/60)<8.5,
                           'watching_tv': lambda t:(t/60)>9 and (t/60)<9.5, 
                           }
        else:
            self.activities = {
                           'brushing_teeth': lambda t:(t/60)>8 and (t/60)<8.5, 
                           }
        self.left_house = False

    def __call__(self, t_mins):
        for act in self.activities:
            doing = self.activities[act](t_mins)
            if doing:
                return act
        return None

    def update_distributions(self, t_mins, activity):
        del self.activities[activity]
        pass

    def does(self, act):
        return act in self.activities
    
    def valid_end(self, endtime, failure):
        return True


class ScheduleSamplerBreakfast():
    def __init__(self):
        self.activities = ['breakfast_food', 'breakfast_beverage']
        self.idle_prob = 0.6
        self.left_house = False

    def __call__(self, t_mins):
        if random.random() < self.idle_prob or len(self.activities) == 0: return None
        activity = random.choice(self.activities)
        return activity

    def update_distributions(self, t_mins, activity):
        self.activities.remove(activity)

    def get_probability(self, activity, t_mins):
        return (1-self.idle_prob)/len(self.activities)


class ScheduleSamplerBreakfastOrLeave():
    def __init__(self):
        self.having_breakfast = random.random() < 0.5
        self.activity_list = {'breakfast_food': {'prob':lambda t:max(min(abs(t-6*60), abs(t-12*60))/60 - 2, 0)*(3/3)}, 
                              'leaving_home_fast':{'prob':lambda t:max(abs(t-12*60)/60 - 3, 0)*(2/9)}, 
                              }
        self.activities = {}
        if self.having_breakfast:
            self.activities['breakfast_food'] = self.activity_list['breakfast_food']
        else:
            self.activities['leaving_home_fast'] = self.activity_list['leaving_home_fast']
        self.max_prob = 2.5
        self.left_house = True

    def __call__(self, t_mins):
        sample = random.random() * self.max_prob
        activity = None
        for act in self.activities:
            thresh = self.get_probability(act, t_mins)
            if thresh > sample:
                activity = act
                self.left_house = False
                break
            sample -= thresh
        return activity

    def update_distributions(self, t_mins, activity):
        self.activities[activity]= {'prob':lambda t:0}

    def get_probability(self, activity, t_mins):
        return self.activities[activity]['prob'](t_mins)

    
    def plot(self, dirname = None):
        start_times = np.arange(6*60,12*60,10)
        fig, axs_numbered = plt.subplots(2,1)
        fig.set_size_inches(27, 18.5)
        axs = {'breakfast_food':axs_numbered[0], 'leaving_home_fast':axs_numbered[1]}
        
        for time in start_times:
            for act, plot in axs.items():
                plot.bar(time/60, self.activity_list[act]['prob'](time), color = color_map[act])
        
        for act, plot in axs.items():
            plot.set_xticks(np.arange(6,12))
            plot.set_xticklabels([str(s)+':00' for s in np.arange(6,12)])
            plot.set_title(act)

        plt.legend()
        if dirname is not None:
            fig.savefig(os.path.join(dirname, 'sampling_distribution_separated.jpg'))
        else:
            plt.show()




class ScheduleSamplerPhoneInterruption():
    def __init__(self):
        self.having_breakfast = random.random() < 0.5
        self.activity_list = {'breakfast_food': {'prob':lambda t:max(min(abs(t-6*60), abs(t-12*60))/60 - 1, 0)*(1/2)}, 
                              'breakfast_beverage': {'prob':lambda t:max(min(abs(t-6*60), abs(t-12*60))/60 - 1, 0)*(1/4)}, 
                              'talk_on_phone':{'prob':lambda t:0.5}, 
                              }
        self.activities = self.activity_list
        self.max_prob = 2.0
        self.left_house = True

    def __call__(self, t_mins):
        sample = random.random() * self.max_prob
        activity = None
        for act in self.activities:
            thresh = self.get_probability(act, t_mins)
            if thresh > sample:
                activity = act
                self.left_house = False
                break
            sample -= thresh
        return activity

    def update_distributions(self, t_mins, activity):
        if activity != 'talk_on_phone':
            self.activities[activity]= {'prob':lambda t:0}

    def get_probability(self, activity, t_mins):
        return self.activities[activity]['prob'](t_mins)

    
    def plot(self, dirname = None):
        start_times = np.arange(6*60,12*60,10)
        fig, axs_numbered = plt.subplots(3,1)
        fig.set_size_inches(27, 18.5)
        axs = {'breakfast_food':axs_numbered[0], 'breakfast_beverage':axs_numbered[1], 'talk_on_phone':axs_numbered[2]}
        
        for time in start_times:
            for act, plot in axs.items():
                plot.bar(time/60, self.activity_list[act]['prob'](time), color = color_map[act])
        
        for act, plot in axs.items():
            plot.set_xticks(np.arange(6,12))
            plot.set_xticklabels([str(s)+':00' for s in np.arange(6,12)])
            plot.set_title(act)

        plt.legend()
        if dirname is not None:
            fig.savefig(os.path.join(dirname, 'sampling_distribution_separated.jpg'))
        else:
            plt.show()

