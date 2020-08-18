import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from bokeh.models import ColumnDataSource, CDSView, GroupFilter
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
flavors = whisky.iloc[:, 2:14]
corr_flavors = pd.DataFrame.corr(flavors)
# plt.figure(figsize=(10,10))
# plt.pcolor(corr_flavors)
# plt.colorbar()

corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.axis("tight")

model = SpectralCoclustering(n_clusters = 6, random_state = 0)
model.fit(corr_whisky)
whisky["Group"] = pd.Series(model.row_labels_, index = whisky.index)
whisky = whisky.loc[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop = True)
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)
plt.figure(figsize = (14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
data = pd.Series([1,2,3,4])
data = data.loc[[3,0,1,2]]
data = data.reset_index(drop=True)

whisky = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@whiskies.csv", index_col=0)
correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)
# First, we import a tool to allow text to pop up on a plot when the cursor
# hovers over it.  Also, we import a data structure used to store arguments
# of what to plot in Bokeh.  Finally, we will use numpy for this section as well!

# from bokeh.models import HoverTool, ColumnDataSource

# # Let's plot a simple 5x5 grid of squares, alternating between two colors.
# plot_values = [1,2,3,4,5]
# plot_colors = ['#0173b2', '#de8f05']

# # How do we tell Bokeh to plot each point in a grid?  Let's use a function that
# # finds each combination of values from 1-5.
# from itertools import product

# grid = list(product(plot_values, plot_values))
# print(grid)
# [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
# # The first value is the x coordinate, and the second value is the y coordinate.

# # Let's store these in separate lists.
# xs, ys = zip(*grid)
# print(xs)
# print(ys)
# (1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5)
# (1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

# # Now we will make a list of colors, alternating between the two chosen colors.
# colors = [plot_colors[i%2] for i in range(len(grid))]
# print(colors)
# ['#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2']

# # Finally, let's determine the strength of transparency (alpha) for each point,
# # where 0 is completely transparent.
# alphas = np.linspace(0, 1, len(grid))

# # Bokeh likes each of these to be stored in a special dataframe, called
# # ColumnDataSource.  Let's store our coordinates, colors, and alpha values.
# source = ColumnDataSource(
#     data = {
#         "x": xs,
#         "y": ys,
#         "colors": colors,
#         "alphas": alphas,
#     }
# )

# # We are ready to make our interactive Bokeh plot!
# from bokeh.plotting import figure, output_file, show

# output_file("Basic_Example.html", title="Basic Example")
# fig = figure(tools="hover")
# fig.rect("x", "y", 0.9, 0.9, source=source, color="colors",alpha="alphas")
# hover = fig.select(dict(type=HoverTool))
# hover.tooltips = {
    # "Value": "@x, @y",
    # }
#show.fig()
cluster_colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161']
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]
region_colors = dict(zip(regions, cluster_colors))
distilleries = list(whisky.Distillery)
correlation_colors = []
for i in range(len(distilleries)):
    for j in range(len(distilleries)):
        if correlations[i,j] < 0.7:                    # if low correlation,
            correlation_colors.append('white')         # just use white.
        else:                                          # otherwise,
            if whisky.Group[i] == whisky.Group[j]:                 # if the groups match,
                correlation_colors.append(cluster_colors[whisky.Group[i]]) # color them by their mutual group.
            else:                                      # otherwise
                correlation_colors.append('lightgray') # color them lightgray.

source = ColumnDataSource(
    data = {
        "x": np.repeat(distilleries,len(distilleries)),
        "y": list(distilleries)*len(distilleries),
        "colors": correlation_colors,
        "correlations": correlations.flatten()
    }
)

output_file("Whisky Correlations.html", title="Whisky Correlations")
fig = figure(title="Whisky Correlations",
    x_axis_location="above", x_range=list(reversed(distilleries)), y_range=distilleries,
    tools="hover,box_zoom,reset")
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = "5pt"
fig.xaxis.major_label_orientation = np.pi / 3
fig.rect('x', 'y', .9, .9, source=source,
     color='colors', alpha='correlations')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Whiskies": "@x, @y",
    "Correlation": "@correlations",
}
points = [(0,0), (1,2), (3,1)]
xs, ys = zip(*points)
colors = ['#0173b2', '#de8f05', '#029e73']

output_file("Spatial_Example.html", title="Regional Example")
location_source = ColumnDataSource(
    data={
        "x": xs,
        "y": ys,
        "colors": colors,
    }
)

fig = figure(title = "Title",
    x_axis_location = "above", tools="hover, save")
fig.plot_width  = 300
fig.plot_height = 380
fig.circle("x", "y", size=10, source=location_source,
     color='colors', line_color = None)

hover = fig.select(dict(type = HoverTool))
hover.tooltips = {
    "Location": "(@x, @y)"
}
def location_plot(title, colors):
    output_file(title+".html")
    location_source = ColumnDataSource(
        data = {
            "x": whisky[" Latitude"],
            "y": whisky[" Longitude"],
            "colors": colors,
            "regions": whisky.Region,
            "distilleries": whisky.Distillery
        }
    )

    fig = figure(title = title,
        x_axis_location = "above", tools="hover, save")
    fig.plot_width  = 400
    fig.plot_height = 500
    fig.circle("x", "y", size=9, source=location_source,
         color='colors', line_color = None)
    fig.xaxis.major_label_orientation = np.pi / 3
    hover = fig.select(dict(type = HoverTool))
    hover.tooltips = {
        "Distillery": "@distilleries",
        "Location": "(@x, @y)"
    }
    show(fig)
    
region_cols = [region_colors[i] for i in list(whisky["Region"])]
classification_cols = [cluster_colors[i] for i in list(whisky["Group"])]

location_plot("Whisky Locations and Regions", region_cols)
location_plot("Whisky Locations and Groups", classification_cols)

