import json
import os
import random
from math import floor
from turtle import write
import matplotlib.pyplot as plt


weekday_schedules = []
weekend_schedules = []
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
"hand_wash_clothes" : "laundry",
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
"diary_journaling" : None,
"wake_up" : None,
"sleep" : None,
}
start_times = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

see_activities = ["brushing_teeth", "showering", "breakfast", "lunch", "dinner", "taking_medication", "getting_dressed", "computer_work", "leaving_home_and_coming_back", "laundry", "socializing"]
see_activities = list(set(activity_map.values()))
see_activities.remove(None)

class ScheduleSampler():
    def __init__(self, data_dir='data/AMT_Schedules', write_to_file=False, filter_num = 3, idle_sampling_factor = 1.0):
        self.activity_histograms = {k:{s:0 for s in start_times} for k in activity_map.values()}
        self.schedule_num = 0
        for root, dirs, files in os.walk(data_dir):
            fig, self.ind_plot = plt.subplots(len(files)-4, len(see_activities), sharey=True, sharex=True)
            fig.set_size_inches(50, 25)
            self.scene_histogram = [None] * len(files)
            for f in files:
                if f in ['A3EF9IKIU8IAHY.json', 'A3FKOG3VU5MP0X.json','AFER49XXMM4PE.json','A332L3FO6RCY6G.json']:
                    continue
                if f.endswith('.json'):
                    with open(os.path.join(root,f)) as file:
                        sch = json.load(file)
                    schedule_processed = self._get_schedule(sch['activitiesChosen'], int(sch["questionnaireData"]["age"][:2]),sch["questionnaireData"]["gender"])
                    if write_to_file: 
                        with open(os.path.join('data/sourcedSchedules/weekday',f), 'w') as f:
                            json.dump(schedule_processed, f)
                    # self.ind_plot[self.schedule_num,0].set_ylabel(f[:6])
                    self.schedule_num += 1
        for i, act in enumerate(see_activities):
            self.ind_plot[0,i].set_title(act)
        running_sum = [0 for _ in start_times]
        self.activity_threshold = {s:{} for s in start_times}
        fig.tight_layout()
        # plt.rcParams["figure.figsize"] = (27, 18.5)
        self.filter_num = filter_num
        for act, freq in self.activity_histograms.items():
            if act is None:
                continue
            filtered = [max(0,f-self.filter_num) for f in freq.values()]
            # plt.bar(freq.keys(), filtered, label=act, bottom=running_sum)
            for i,st in enumerate(freq.keys()):
                self.activity_threshold[st][act] = (running_sum[i], running_sum[i] + filtered[i])
            running_sum = [r+f for r,f in zip(running_sum, filtered)]
            # plt.plot(freq.keys(), [activity_threshold[s][act] for s in start_times])
        # plt.legend()
        self.sampling_range = max(running_sum) * idle_sampling_factor
        self.removed_activities = []
    
    def _get_schedule(self, data, age, gender):
        if gender == 'Female': c = [age/60, 0, 0]
        if gender == 'Male': c = [0, 0, age/60]
        # self.ind_plot[self.schedule_num,0].set_ylabel(str(age) + ', '+gender[0])
        activity_times = {}
        self.scene_histogram[self.schedule_num] = {k:{s:0 for s in start_times} for k in see_activities}
        for timestring,activities in data.items():
            start_time = timestring.split('m')[0]
            start_time = int(start_time[:-1]) if start_time[-1] == 'a' else int(start_time[:-1])+12
            if start_time == 24:
                start_time = 12
            for act in set(activities):
                sch_activity = activity_map[act]
                if sch_activity is None:
                    continue
                self.activity_histograms[sch_activity][start_time] += 1
                self.scene_histogram[self.schedule_num][sch_activity][start_time] += 1
                if sch_activity in activity_times:
                    activity_times[sch_activity].append(start_time)
                else:
                    activity_times[sch_activity] = [start_time]
        for i, act in enumerate(see_activities):
            hist = self.scene_histogram[self.schedule_num][act]
            self.ind_plot[self.schedule_num][i].bar(hist.keys(), hist.values(), color=c)
            self.ind_plot[self.schedule_num][i].set_xticks([6,9,12,15,18,21,24])
            self.ind_plot[self.schedule_num][i].set_xticklabels(['6:00','','12:00','','18:00','','24:00'])
        schedule = {}
        for sch_activity, act_st in activity_times.items(): 
            schedule[sch_activity] = [[act_st[0], act_st[0]+1]]
            for time in act_st[1:]:
                if schedule[sch_activity][-1][1] == time:
                    schedule[sch_activity][-1][1] = time+1
                else:
                    schedule[sch_activity].append([time, time+1])
        return schedule

    # def plot(self):
    #     self.fig.show()

    def __call__(self, t_mins, remove=False):
        sample = random.random()*self.sampling_range
        activity = None
        st = int(floor(t_mins/60))
        for act, thresh in self.activity_threshold[st].items():
            if thresh[0] < sample and thresh[1] >= sample:
                activity = act
                break
        if activity in self.removed_activities:
            return None
        else:
            if remove: self.remove(activity)
            return activity

    def remove(self, activity):
        self.removed_activities.append(activity)

    def sample_time_for(self, activity):
        max_sample = sum(self.activity_histograms[activity].values()) - len(self.activity_histograms[activity])*self.filter_num
        sample = random.random()*max_sample
        for st, freq in self.activity_histograms[activity].items():
            filtered_freq = freq-self.filter_num
            if sample <= filtered_freq:
                t_hours = st + random.random()
                t_mins = int(round(t_hours*60))
                return t_mins
            sample -= filtered_freq
        raise RuntimeError('This should not happen!')

    def reset(self):
        self.removed_activities = []
