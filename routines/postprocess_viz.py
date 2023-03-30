# %%
import argparse
from copy import deepcopy
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

from ScheduleDistributionSampler import color_map

min_time = 360
plt.rcParams.update({'font.size': 22})

grey = sns.color_palette(palette='pastel')[7]

greyed_activities = ["brushing_teeth", "showering", "leaving_home_fast", "watching_tv", "talk_on_phone", "online_meeting", "going_to_the_bathroom", "listening_to_music"]

def alpha_act(activity):
    # if activity == "dinner":
    return 0.5
    # return 0.3

color_map_trunc = deepcopy(color_map)
# color_map_trunc["vaccuum_cleaning"] = None 
# color_map_trunc["reading"] = None 
# color_map_trunc["going_to_the_bathroom"] = None 
# color_map_trunc["getting_dressed"] = None 
# color_map_trunc["kitchen_cleaning"] = None 
# color_map_trunc["take_out_trash"] = None 
# color_map_trunc["wash_dishes"] = None 
# color_map_trunc["playing_music"] = None 
# color_map_trunc["listening_to_music"] = None


def time_human(time_mins):
    time_mins = int(round(time_mins))
    mins = time_mins%60
    time_mins = time_mins//60
    hrs = time_mins%24
    time_mins = time_mins//24
    days = time_mins
    h = '{:02d}:{:02d}'.format(hrs,mins)
    # if days != 0:
    #     h = str(days)+'day - '+h
    return h

# %%

def dump_visuals(root_dir):
    print(os.listdir(root_dir))

    for ind in os.listdir(root_dir):
        directory = os.path.join(root_dir, ind)


        def to_mins(str, add_day=False):
            hrs_mins = [int(n) for n in str.split(':')]
            total = int(60*hrs_mins[0])+hrs_mins[1]
            if add_day: total += 24*60
            return total

        train_dirs = [os.path.join(directory,'scripts_train',fn) for fn in os.listdir(os.path.join(directory,'scripts_train'))]
        train_dirs.sort()
        test_dirs = [os.path.join(directory,'scripts_test',fn) for fn in os.listdir(os.path.join(directory,'scripts_test'))]
        test_dirs.sort()

        for colors in ['color','greyed']:
            local_color_map = deepcopy(color_map)
            if colors == 'greyed':
                for act in greyed_activities:
                    local_color_map[act] = grey
            fig,ax = plt.subplots()
            fig.set_size_inches(30,15)
            activities_labeled = []
            last_time = None
            for sch_cnt, file in enumerate(train_dirs+test_dirs):
                header = True
                movements = []
                movement_magns = []
                for line in open(file).readlines():
                    ## Break when the header is done
                    if header and line.strip() == '':
                        header = False
                        continue
                    elif header:
                        activity = line[:line.index('(')].strip()
                        times = line[line.index('(')+1:line.index(')')].strip().split('-')
                        ## Break when the activity starts past midnight
                        if len(times) > 3: break
                        if len(times) > 2:
                            assert times[1].strip() == '1day'
                        start = to_mins(times[0].strip())
                        end = to_mins(times[-1].strip(), add_day=len(times)>2)
                        end = min(end, 24*60)
                        if activity not in activities_labeled:
                            ax.barh(sch_cnt+1, end-start, align='center', left=start, label=activity, color=local_color_map[activity], alpha=alpha_act(activity))
                            activities_labeled.append(activity)
                        else:
                            ax.barh(sch_cnt+1, end-start, align='center', left=start, color=local_color_map[activity], alpha=alpha_act(activity))
                    else:
                        if 'moved' in line:
                            time_idx = line.index(':')
                            movement = float(line[:time_idx])
                            movement_magn = line.count(',')+1
                            ax.plot(movement, sch_cnt+1, '.', c=[1,1,1,1], markersize=movement_magn*4)
                        try:
                            last_time = float(line.strip())
                        except:
                            pass
                        if "(s)" in line and "]->[" in line and 'OPEN' in line[line.index('>'):]:
                            ax.plot(last_time, sch_cnt+1, 'tab:blue', markersize=10, marker='|')
                        elif "(s)" in line and "]->[" in line and 'CLOSED' in line[line.index('>'):]:
                            ax.plot(last_time, sch_cnt+1, 'tab:red', markersize=10, marker='|')



                # _ = ax.set_yticks([])
                try:
                    start_time = json.load(open(os.path.join(directory,'info.json')))['start_time']
                    end_time = json.load(open(os.path.join(directory,'info.json')))['end_time']
                except:
                    start_time = 6*60
                    end_time = 24*60
                ax.set_xlim([start_time,end_time+(end_time-start_time)/4])
                steps = 60 if end_time-start_time < 360 else 3*60
                _ = ax.set_xticks(np.arange(start_time, end_time+1, steps))
                _ = ax.set_xticklabels([time_human(t) for t in np.arange(start_time, end_time+1, steps)], fontsize=45)

            fig.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(directory,f'schedules_{colors}.jpg'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/OrderlyHouseholds0308', help='Directory to output data in')
    parser.add_argument('--move_visuals', action='store_true', default=False, help='Set this to generate a complete dataset of all individuals and personas')

    args = parser.parse_args()

    dump_visuals(args.path)

    assert os.path.exists(args.path)

    if args.move_visuals:
        for f in glob.glob(args.path+'/*/*.jpg') + glob.glob(args.path+'/*/*.jpeg'):
            target_file = f.replace('sourcedRoutines','dataVisuals').replace('/schedule','_schedule').replace('/sampling','_sampling')
            if 'schedule' in f or 'sampling' in f:
                if not os.path.exists(os.path.dirname(target_file)): os.makedirs(os.path.dirname(target_file))
                shutil.copy(f, target_file)
