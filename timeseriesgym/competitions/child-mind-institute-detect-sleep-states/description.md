# Child Mind Institute - Detect Sleep States

[Competition Link](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview)

## Overview

Your work will improve researchers' ability to analyze accelerometer data for sleep monitoring and enable them to conduct large-scale studies of sleep. Ultimately, the work of this competition could improve awareness and guidance surrounding the importance of sleep. The valuable insights into how environmental factors impact sleep, mood, and behavior can inform the development of personalized interventions and support systems tailored to the unique needs of each child.

## Description

### Goal of the Competition

The goal of this competition is to detect sleep onset and wake. You will develop a model trained on wrist-worn accelerometer data in order to determine a person's sleep state.

Your work could make it possible for researchers to conduct more reliable, larger-scale sleep studies across a range of populations and contexts. The results of such studies could provide even more information about sleep.

The successful outcome of this competition can also have significant implications for children and youth, especially those with mood and behavior difficulties. Sleep is crucial in regulating mood, emotions, and behavior in individuals of all ages, particularly children. By accurately detecting periods of sleep and wakefulness from wrist-worn accelerometer data, researchers can gain a deeper understanding of sleep patterns and better understand disturbances in children.

### Context

The “Zzzs” you catch each night are crucial for your overall health. Sleep affects everything from your development to cognitive functioning. Even so, research into sleep has proved challenging, due to the lack of naturalistic data capture alongside accurate annotation. If data science could help researchers better analyze wrist-worn accelerometer data for sleep monitoring, sleep experts could more easily conduct large-scale studies of sleep, thus improving the understanding of sleep's importance and function.

Current approaches for annotating sleep data include sleep logs, which are the gold standard for detecting the onset of sleep. However, they are impractical for many participants to use reliably, and fail to capture the nuanced difference between heading to bed and falling asleep (or, conversely, waking up and getting out of bed). Heuristic-based software is another solution that attempts to identify sleep windows, though these rely on human-engineered features of sleep (i.e. arm angle) that vary across individuals and don’t accurately summarize the sleep windows that experts can visually detect from their data. With improved tools to analyze sleep data on a large scale, researchers can explore the relationship between sleep and mood/behavioral difficulties. This knowledge can lead to more targeted interventions and treatment strategies.

Competition host Child Mind Institute (CMI) transforms the lives of children and families struggling with mental health and learning disorders by giving them the help they need. CMI has become the leading independent nonprofit in children’s mental health by providing gold-standard evidence-based care, delivering educational resources to millions of families each year, training educators in underserved communities, and developing tomorrow’s breakthrough treatments.

Established with a foundational grant from the Stavros Niarchos Foundation (SNF), the SNF Global Center for Child and Adolescent Mental Health at the Child Mind Institute works to accelerate global collaboration on under-researched areas of children’s mental health and expand worldwide access to culturally appropriate trainings, resources, and treatment. A major goal of the SNF Global Center is to expand innovations in clinical assessment and intervention, to include building, testing, and deploying new technologies to augment mental health care and research, including mobile apps, sensors, and analytical tools.

Your work will improve researchers' ability to analyze accelerometer data for sleep monitoring and enable them to conduct large-scale studies of sleep. Ultimately, the work of this competition could improve awareness and guidance surrounding the importance of sleep. The valuable insights into how environmental factors impact sleep, mood, and behavior can inform the development of personalized interventions and support systems tailored to the unique needs of each child.

### Acknowledgement

The data used for this competition was provided by the [Healthy Brain Network](https://healthybrainnetwork.org/), a landmark mental health study based in New York City that will help children around the world. In the Healthy Brain Network, families, community leaders, and supporters are partnering with the Child Mind Institute to unlock the secrets of the developing brain. In addition to generous support provided by the Kaggle team, financial support has been provided by the Stavros Niarchos Foundation (SNF) as part of its Global Health Initiative (GHI) through the SNF Global Center for Child and Adolescent Mental Health at the Child Mind Institute.

## Evaluation

Submissions are evaluated on the average precision of detected events, averaged over timestamp error tolerance thresholds, averaged over event classes.

Detections are matched to ground-truth events within error tolerances, with ambiguities resolved in order of decreasing confidence. For both event classes, we use error tolerance thresholds of 1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30 in minutes, or 12, 36, 60, 90, 120, 150, 180, 240, 300, 360 in steps.

You may find Python code for this metric in the `event_detection_average_precision.py` file.

### Detailed Description

Evaluation proceeds in three steps:
1. **Assignment** - Predicted events are matched with ground-truth events
2. **Scoring** - Each group of predictions is scored against its corresponding group of ground-truth events via Average Precision (AP)
3. **Reduction** - The multiple AP scores are averaged to produce a single overall score

### Assignment

For each set of predictions and ground-truths within the same `event x tolerance x series_id` group, we **match** each ground-truth to the highest-confidence unmatched prediction occurring within the allowed tolerance.

Some ground-truths may not be matched to a prediction and some predictions may not be matched to a ground-truth. They will still be accounted for in the scoring, however.

### Scoring

Collecting the events within each series_id, we compute an **Average Precision (AP)** score for each `event x tolerance group`. The average precision score is the area under the precision-recall curve generated by decreasing confidence score thresholds over the predictions. In this calculation, matched predictions over the threshold are scored as TP and unmatched predictions as FP. Unmatched ground-truths are scored as FN.

### Reduction
The final score is the average of the above AP scores, first averaged over tolerance, then over event.

## Submission File

For each series indicated by `series_id`, predict each event occurring in that series by giving the event type and the step where the event occurred as well as a confidence score for that event. The evaluation metric additionally requires a `row_id` with an enumeration of events.

The file should contain a header and have the following format:

```bash
row_id,series_id,step,event,score
0,038441c925bb,100,onset,0.0
1,038441c925bb,105,wakeup,0.0
2,03d92c9f6f8a,80,onset,0.5
3,03d92c9f6f8a,110,wakeup,0.5
4,0402a003dae9,90,onset,1.0
5,0402a003dae9,120,wakeup,1.0
...
```

Note that while the ground-truth annotations were made following certain conventions (as described on the Data page), there are no such restrictions on your submission file.

## Prizes

- 1st Place - $15,000
- 2nd Place - $10,000
- 3rd Place - $8,000
- 4th Place - $7,000
- 5th Place - $5,000
- 6th Place - $5,000

## Dataset Description

The dataset comprises about 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: *onset*, the beginning of sleep, and *wakeup*, the end of sleep. Your task is to detect the occurrence of these two events in the accelerometer series.

While sleep logbooks remain the gold-standard, when working with accelerometer data we refer to sleep as the longest single period of inactivity while the watch is being worn. For this data, we have guided raters with several concrete instructions:

- A single sleep period must be at least 30 minutes in length
- A single sleep period can be interrupted by bouts of activity that do not exceed 30 consecutive minutes
- No sleep windows can be detected unless the watch is deemed to be worn for the duration (elaborated upon, below)
- The longest sleep window during the night is the only one which is recorded
- If no valid sleep window is identifiable, neither an onset nor a wakeup event is recorded for that night.
- Sleep events do not need to straddle the day-line, and therefore there is no hard rule defining how many may occur within a given period. However, no more than one window should be assigned per night. For example, it is valid for an individual to have a sleep window from 01h00–06h00 and 19h00–23h30 in the same calendar day, though assigned to consecutive nights
- There are roughly as many nights recorded for a series as there are 24-hour periods in that series.

Though each series is a continuous recording, there may be periods in the series when the accelerometer device was removed. These period are determined as those where suspiciously little variation in the accelerometer signals occur over an extended period of time, which is unrealistic for typical human participants. Events are not annotated for these periods, and you should attempt to refrain from making event predictions during these periods: an event prediction will be scored as false positive.

Each data series represents this continuous (multi-day/event) recording for a unique experimental subject.

### Files and Field Descriptions

- **train_series.parquet** - Series to be used as training data. Each series is a continuous recording of accelerometer data for a single subject spanning many days.
    - `series_id` - Unique identifier for each accelerometer series.
    - `step` - An integer timestep for each observation within a series.
    - `timestamp` - A corresponding datetime with ISO 8601 format %Y-%m-%dT%H:%M:%S%z.
    - `anglez` - As calculated and described by the [GGIR package](https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html#4_Inspecting_the_results), z-angle is a metric derived from individual accelerometer components that is commonly used in sleep detection, and refers to the angle of the arm relative to the vertical axis of the body
    - `enmo` - As calculated and described by the [GGIR package](https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html#4_Inspecting_the_results), ENMO is the Euclidean Norm Minus One of all accelerometer signals, with negative values rounded to zero. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features
- **test_series.parquet** - Series to be used as the test data, containing the same fields as above. You will predict event occurrences for series in this file.
- **train_events.csv** - Sleep logs for series in the training set recording onset and wake events.
    - `series_id` - Unique identifier for each series of accelerometer data in train_series.parquet.
    - `night` - An enumeration of potential onset / wakeup event pairs. At most one pair of events can occur for each night.
    - `event` - The type of event, whether onset or wakeup.
    - `step` and `timestamp` - The recorded time of occurence of the event in the accelerometer series.
- **sample_submission.csv** - A sample submission file in the correct format. See the **Evaluation** tab for more details.
