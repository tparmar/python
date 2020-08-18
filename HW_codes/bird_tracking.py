import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

birddata = pd.read_csv("bird_tracking.txt")
bird_names = pd.unique(birddata.bird_name)
plt.figure(figsize = (7,7))
for bird_name in bird_names:
    ix = birddata.bird_name == bird_name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    plt.plot(x,y, ".", label = bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc = "lower right")
ix = birddata.bird_name == bird_name
speed = birddata.speed_2d[ix]
ind = np.isnan(speed)
plt.hist(speed[~ind], bins = np.linspace(0,30,20))
plt.xlabel("2D speed(m/s)")
plt.ylabel("Frequency")
date_str = birddata.date_time[0]
datetime.datetime.strptime(date_str[:-3], "%Y-%m-%d %H:%M:%S")
timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
        (birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))
birddata["timestamp"] = pd.Series(timestamps, index = birddata.index)
birddata.timestamp[4] - birddata.timestamp[3]
times = birddata.timestamp[birddata.bird_name == "Eric"]
elapsed_time = [time-times[0] for time in times]
plt.plot(np.array(elapsed_time) / datetime.timedelta(days=1))
plt.xlabel("Observation")
plt.ylabel("Elapsed time(days)")
data = birddata[birddata.bird_name == "Eric"]
times = data.timestamp
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days = 1)
next_day = 1
inds = []
daily_mean_speed = []
for (i, t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []
plt.figure(figsize = (8,6))
plt.plot(daily_mean_speed)
plt.xlabel("Day")
plt.ylabel("Mean speed (m/s)")