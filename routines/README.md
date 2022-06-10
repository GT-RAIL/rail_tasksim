# Household Object Movements from Everyday Routines (HOMER)

This is the home of the HOMER dataset generator. This codebase allows creation of new datasets in addition to the five persona datasets which can be found [here](). To generate the complete object arrangements, we use the [VirtualHome](http://virtual-home.org) simulator.

To generate a dataset use `SampleRoutinesFromData.py` with the appropriate arguments. To replicate the existing dataset, use  `python3 SampleRoutinesFromData.py --sampler=persona`. If you provide a specific persona ID ('persona0', 'persona1', ... 'persona4') or an individual ID from data/AMT_Schedules', the Sampler will generate a dataset pertaining to that distribution. If you only mention 'persona' or 'individual', the sampler loops through all the 'persona's or 'individual's.


# Data generation process

<img src="data/personaBasedSchedules/visuals/HOMERtitle.jpg">

This page is a walk-through of the complete process used to generate the HOMER dataset. The HOMER dataset is built on the [VirtualHome](http://virtual-home.org) simulator. The published dataset is composed of routine behaviors spanning several weeks for five personas in an apartment setting with four rooms containing 66-77 objects and 33 atomic actions such as *find*, *walk*, *grab*, etc. 

We first make a list of Activities of Daily Living related to in-home routines, and then source our dataset using a two-tier strategy, separately sourcing high level activity schedules comprising of the above activities, and the low level action sequences to perform each activity. The 22 high-level activities of daily living we use are as follows:

* bathe or shower 
* brush teeth 
* clean 
* clean kitchen 
* come home 
* computer work 
* connect with friends 
* do laundry 
* get dressed 
* leave home 
* listen to music 
* play music 
* prepare and eat breakfast 
* prepare and eat dinner 
* prepare and eat lunch 
* read 
* take medication 
* take out trash 
* use restroom 
* vacuum clean 
* wash dishes 
* watch TV 


In the following sections, we first walk through the process of sourcing and processing each of the Acitivty Schedules and Activity Scripts separately, and then explain how the two are combined and used to sample the final routines.

## Activity Schedules

### Original data from AMT

To obtain realistic activity schedules, we first asked workers on Amazon Mechanical Turk about which hours on a typical day they are likely to be doing each activity. The prompt was as follows
<img src="data/personaBasedSchedules/visuals/AMTprompt.png" width="800" align="top">.....and so on till midnight

We obtained this data from 25 workers, of which we filter out 4 for providing nonsensical schedules such as brushing and having dinner constantly throughout the day. The remaining 21 candidates include 4 female and 17 male participants, 8 participants aged 25-35, 7 aged 35-45, 3 aged 45-55, and 3 aged over 55. We also reclassified meals so that those happening in the morning were classified as breakfast (even if the raw input marked them as dinner), those in the afternoon were lunch, and in the evening were dinner. The following figure shows the filtered input with each row being one individual's input and each column signifying an acitivity.

<img src="data/personaBasedSchedules/visuals/correctedSchedules.png">


### Clustering activity samples into habits


The original samples are noisy and include idiosyncratic preferences of the individuals, so we cluster the samples for each activity to obtain aggregate distributions representing a common underlying habit of the individual samples that form the cluster. We represent each sample using a feature vector containing 
* the means of a fitted mixture of two gaussians, and 
* the number of hours that activity is done in the morning (before 12:00), in the afternoon (between 12:00 and 18:00), and in the evening (after 18:00). 

Using these features to represent every sample, we divide the samples for each activity into 4 clusters using k-means clustering. We remove the clusters containing fewer than 3 samples. This results in upto 4 aggregate distributions representing different habits pertaining to that activity. The following plots represent the  aggregate distributions for each cluster, the number of samples belonging to that cluster out of the 21 initial responses, and the original responses.

<img src="data/personaBasedSchedules/visuals/histograms/brushing_teeth.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/showering.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/breakfast.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/lunch.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/dinner.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/vaccuum_cleaning.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/getting_dressed.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/leave_home.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/come_home.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/listening_to_music.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/playing_music.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/reading.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/watching_tv.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/going_to_the_bathroom.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/cleaning.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/laundry.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/take_out_trash.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/wash_dishes.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/kitchen_cleaning.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/taking_medication.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/computer_work.jpg" width="700" align="top">
<img src="data/personaBasedSchedules/visuals/histograms/socializing.jpg" width="700" align="top">

As seen in these figures, the clusters do represent semantically meaningful habits, like *brushing teeth in the morning* v.s. *brushing teeth twice a day*, and having an *early breafast* v.s. a *late breakfast* v.s. *skipping breakfast*.

### Composing personas using habits

To compose complete temporal activity distributions representing fictitious persona, we combine a habit for each activity. We implicitly assume the different habits for the activities to be independent and ensure that our personas are distinct by using genetic optimization to maximize their average pairwise KL-divergence using genetic optimisation. The resulting five personas have the following activity distributions.

**Persona 0**

<img src="data/dataVisuals/persons0509/persona0_sampling_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 1**

<img src="data/dataVisuals/persons0509/persona1_sampling_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 2**

<img src="data/dataVisuals/persons0509/persona2_sampling_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 3**

<img src="data/dataVisuals/persons0509/persona3_sampling_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 4**

<img src="data/dataVisuals/persons0509/persona4_sampling_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

## Activity Scripts

We ask participants to emulate each activity from the above list on a simulator. We used the [VirtualHome simulator](http://virtual-home.org), as it supports human agents, object interaction, and high-level semantic commands without requiring low-level motion control. We recruited 23 participants to compose step-by-step action sequences for each activity, defining the avatar's movement, interactions with various objects, and the time duration required for it. In this manner, we obtained 61 scripts in total, covering all of our 22 activities. Each script consists of action sequences as well as the estimated minimum and maximum time duration needed to do these actions ('## \<min_time\>-\<max_time\> in the following script snippet'), as shown in the following snippet of an action script representing *brushing teeth*.

```
## 1-2
[Walk] <bathroom>
[Walk] <toothbrush>
[Find] <toothbrush>
[Grab] <toothbrush>
## 0-1
[Walk] <bathroom_cabinet>
[Find] <bathroom_cabinet>
[Open] <bathroom_cabinet>
[Find] <tooth_paste>
[Grab] <tooth_paste>
[Close] <bathroom_cabinet>
[Pour] <tooth_paste> <toothbrush>
[Find] <bathroom_counter>
[Putback] <tooth_paste> <bathroom_counter>
.....
```

## Sampling routines

We use a temporal activity distribution and an action script per activity to completely define each of our five fictitious persona, and use Monte Carlo sampling, detailed in the supplementary text, to generate samples of their daily routines.

Starting at 6am, we sample an activity from the schedule distribution, and obtain an end time for that activity by uniformly sampling in the duration range from the script. We sample another activity from the schedule distribution at that end time. If the same activity is sampled again, the activity is continued, and another end time is sampled between that time and the maximum end time, otherwise the new activity is started. By iteratively sampling activities in this manner until the end time of midnight, we obtain samples of timestamped action sequences representing daily schedules. This process is outlined below.

<img src="data/personaBasedSchedules/visuals/HOMERgen.png" width="700" align="top">



## Final Result

The process outlined on this page finally yields routine samples for all five personas. We generate 60 routines per persona, which are visualized below alongwith the resulting distribution of activities throughout the day.

**Persona 0**

<img src="data/dataVisuals/persons0509/persona0_schedule_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/persona0_schedules.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 1**

<img src="data/dataVisuals/persons0509/persona1_schedule_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/persona1_schedules.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 2**

<img src="data/dataVisuals/persons0509/persona2_schedule_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/persona2_schedules.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 3**

<img src="data/dataVisuals/persons0509/persona3_schedule_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/persona3_schedules.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">

**Persona 4**

<img src="data/dataVisuals/persons0509/persona4_schedule_distribution.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/persona4_schedules.jpg" width="650" align="top"><img src="data/dataVisuals/persons0509/legend.jpeg" width="100" align="top">



