# %%
from ast import dump
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
# from statsmodels.stats.inter_rater import fleiss_kappa
# from scipy.interpolate import make_interp_spline

min_time = 360
plt.rcParams.update({'font.size': 22})

# %%
color_map = {
 "brushing_teeth" : sns.color_palette()[0], 
 "showering" : sns.color_palette()[1], 
 "breakfast" : sns.color_palette()[2], 
 "dinner" : sns.color_palette()[3], 
 "computer_work" : sns.color_palette()[4], 
 "lunch" : sns.color_palette()[5], 
 
 "cleaning" : sns.color_palette(palette='dark')[0], 
 "laundry" : sns.color_palette(palette='dark')[1], 
 "leave_home" : sns.color_palette(palette='dark')[2], 
 "come_home" : sns.color_palette(palette='dark')[2], 
 "socializing" : sns.color_palette(palette='dark')[3], 
 "taking_medication" : sns.color_palette(palette='dark')[4], 
 "watching_tv" : sns.color_palette(palette='dark')[5], 
 "vaccuum_cleaning" : None, #sns.color_palette(palette='dark')[6], 
 "reading" : None, #sns.color_palette(palette='dark')[7],

#  "going_to_the_bathroom" : None, 
#  "getting_dressed" : None, 
#  "kitchen_cleaning" : None, 
#  "take_out_trash" : None, 
#  "wash_dishes" : None, 
#  "playing_music" : None, 
#  "listening_to_music" : None

 
 "going_to_the_bathroom" : sns.color_palette(palette='pastel')[0],
 "getting_dressed" : sns.color_palette(palette='pastel')[1],
 "kitchen_cleaning" : sns.color_palette(palette='pastel')[2],
 "take_out_trash" : sns.color_palette(palette='pastel')[3],
 "wash_dishes" : sns.color_palette(palette='pastel')[4],
 "playing_music" : sns.color_palette(palette='pastel')[5],
 "listening_to_music" : sns.color_palette(palette='pastel')[6]
}

color_map_trunc = color_map
color_map_trunc["vaccuum_cleaning"] = None 
color_map_trunc["reading"] = None 
color_map_trunc["going_to_the_bathroom"] = None 
color_map_trunc["getting_dressed"] = None 
color_map_trunc["kitchen_cleaning"] = None 
color_map_trunc["take_out_trash"] = None 
color_map_trunc["wash_dishes"] = None 
color_map_trunc["playing_music"] = None 
color_map_trunc["listening_to_music"] = None


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

# %%

def dump_visuals(root_dir):
    print(os.listdir(root_dir))
    # %%
    times = np.arange(6*60, 24*60, 10)
    num_routines = 60

    for ind in os.listdir(root_dir):
        directory = os.path.join(root_dir, ind)

        fig,ax = plt.subplots()
        fig.set_size_inches(30,10)


        activity_freq = {t:{k:0 for k in color_map.keys()} for t in times}
        activities_labeled = []

        sch_cnt = 0
        with open(os.path.join(directory, 'script_usage.txt')) as f:
            data = f.readlines()
            for line in data[1:]:
                _, activity, start, end = line.split(';')
                start = float(start.strip())
                end = float(end.strip())
                end = min(end, 24*60)
                for t in times:
                    if t>start and t<end:
                        activity_freq[t][activity] += 1
            bottoms = times*0.0
            for act in color_map.keys():
                freqs = [act_fr[act]/num_routines for act_fr in activity_freq.values()]
                # B_spline_coeff = make_interp_spline(times, freqs)
                # X_Final = np.linspace(times.min(), times.max(), 500)
                # Y_Final = B_spline_coeff(X_Final)
                if color_map_trunc[act] is not None:
                    ax.plot(times, freqs, label=act, color=color_map[act], linewidth=5)
                # ax.bar(times, freqs, bottom=bottoms, label=act, color=color_map[act], width = 6.5)
                bottoms += np.array(freqs)
            misclassification_prob = [min(sum(activity_freq[t].values())/num_routines, 1-max(activity_freq[t].values())/num_routines) for t in times]
            # ax.plot(times, misclassification_prob, '-.k', label='misclassification probability')
            _ = ax.set_ylabel('Probability of Activity', fontsize=40)
            _ = ax.set_xlabel('Time', fontsize=40)
            _ = ax.set_xticks(np.arange(6*60,24*60+1, 3*60))
            _ = ax.set_xticklabels([time_human(t) for t in np.arange(6*60,24*60, 3*60)]+['24:00'], fontsize=35)

        ## fleiss kappa calculation
        all_activities = ['idle']+list(color_map.keys())
        for t in activity_freq:
            activity_freq[t]['idle'] = num_routines - sum(activity_freq[t].values())
        act_fracs = [sum([act_fr[act] for act_fr in activity_freq.values()]) for act in all_activities]
        sum_sq_act_fracs = sum(act_fracs)*sum(act_fracs)
        P_e_bar = sum([a*a/sum_sq_act_fracs for a in act_fracs])
        P_bar = (sum([sum([activity_freq[t][act]*activity_freq[t][act] for act in all_activities]) for t in times]) - (len(times)*num_routines)) / (len(times)*num_routines*(num_routines+1))
        kappa = (P_bar - P_e_bar)/(1 - P_e_bar)
        other_kappa = (P_bar - (1/len(all_activities))) /(1 - (1/len(all_activities)))
        print(ind,kappa, other_kappa)

        avg_miscl_prob = sum(misclassification_prob)/len(misclassification_prob)
        # _ = plt.legend(loc='upper right')
        fig.tight_layout()
        # fig.suptitle(ind+'-- avg. misclassification probability = '+'{:1.3f}'.format(avg_miscl_prob)+' -- Fleiss\' Kappa = '+'{:1.5f}'.format(kappa)+' -- Holly & Guilford\'s G = '+'{:1.5f}'.format(other_kappa))
        plt.savefig(os.path.join(directory,'schedule_distribution_separate.jpg'))

        # with open(os.path.join(directory, 'info.json')) as f:
        #     info = json.load(f)
        # info['misclassification_prob'] = avg_miscl_prob
        # info['fleiss_kappa'] = kappa
        # with open(os.path.join(directory, 'info.json'), 'w') as f:
        #     json.dump(info, f)



        fig,ax = plt.subplots()
        fig.set_size_inches(30,20)

        activities_labeled = []

        # activities = ['idle'] + list(color_map.keys())
        # times = np.arange(6*60,24*60+1,10)
        # fleis_act = np.empty((60, len(times))).astype(int)
        t_idx = 0
        sch_cnt = 0
        with open(os.path.join(directory, 'script_usage.txt')) as f:
            prev_start = min_time
            data = f.readlines()
            for line in data[1:]:
                _, activity, start, end = line.split(';')
                start = float(start.strip())
                end = float(end.strip())
                end = min(end, 24*60)
                if prev_start > end:
                    prev_start = min_time
                    sch_cnt += 1
                    if sch_cnt > 45:
                        break
                    t_idx = 0
                if activity not in activities_labeled:
                    ax.barh(sch_cnt, end-start, align='center', left=start, label=activity, color=color_map[activity])
                    activities_labeled.append(activity)
                ax.barh(sch_cnt, end-start, align='center', left=start, color=color_map[activity])
                prev_start = start
                # while (times[t_idx] < end):
                #     if times[t_idx] < start:
                #         t_idx += 1
                #         continue
                #     # fleis_act[sch_cnt][t_idx] = activities.index(activity)
                #     t_idx += 1

            _ = ax.set_xlabel('Time', fontsize=40)
            _ = ax.set_yticks([])
            _ = ax.set_xticks(np.arange(6*60,24*60+1, 3*60))
            _ = ax.set_xticklabels([time_human(t) for t in np.arange(6*60,24*60, 3*60)]+['24:00'], fontsize=45)

        # np.savetxt('temp.csv',fleis_act.astype(int), delimiter=',')
        # print(fleis_act)

        # kappa = fleiss_kappa(fleis_act)

        # _ = plt.legend(loc='upper right')
        fig.tight_layout()
        # fig.suptitle(ind+' -- Fleiss\' Kappa = '+'{:1.5f}'.format(kappa)+' -- Holly & Guilford\'s G = '+'{:1.5f}'.format(other_kappa))
        plt.savefig(os.path.join(directory,'schedules.jpg'))



dump_visuals('data/sourcedRoutines/persona0329')
# dump_visuals('data/sourcedRoutines/Individual0219')
# dump_visuals('data/sourcedRoutines/completePersona0214')
# dump_visuals('data/sourcedRoutines/completeIndividual0214')

import glob
import shutil

# fig, ax = plt.subplots()
# for act, col in color_map.items():
#     ax.plot(0,0,color=col, label=act, linewidth=20)
#     ax.legend(fontsize=45)
#     fig.set_size_inches(15,35)
#     fig.savefig(os.path.join('data/dataVisuals','legend.jpg'))

for dir in ['persona0329']:
# dir = 'completeIndividual0214'
    for f in glob.glob('data/sourcedRoutines/'+dir+'/*/*.jpg'):
        target_file = f.replace('sourcedRoutines','dataVisuals').replace('/schedule','_schedule')
        if 'schedule' in f:
            if not os.path.exists(os.path.dirname(target_file)): os.makedirs(os.path.dirname(target_file))
            target_file = target_file.replace('/schedule','_schedule')
            shutil.copy(f, target_file)