import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
def distance(p1,p2):   
    return np.sqrt(np.sum(np.power(p1-p2, 2)))
p1 = np.array([1,1])
p2 = np.array([4,4])
def majority_vote(vote):
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] +=1
        else:
            vote_counts[vote] = 1
    winners = []
    max_counts = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_counts:
            winners.append(vote)
    return random.choice(winners)
def majority_vote_short(vote):
    mode, count = ss.mstats.mode(votes)
    return mode
votes = [1,2,3,1,2,3,1,2,3,1,2,3,3,3,3, 2,2,2]
points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3]])
p = np.array([2.5, 2])
def find_nearest_neighbors(p, points, k = 5):    
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[0:k]
ind = find_nearest_neighbors(p, points, 3)
def knn_predict(p, points, outcomes, k=5):
    find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])
def generate_synth_data(n=50):
    r = ss.norm(1,1).rvs(n,2)
    i = ss.norm(0,1).rvs(n,2)
    points = np.concatenate((r,i), axis=0)
    t = np.repeat(0,n)
    y = np.repeat(1,n)
    outcomes = np.concatenate((t,y))
    return (points, outcomes)
n = 20
(points, outcomes) = generate_synth_data(n)
# plt.figure()
# plt.plot(points[:n, 0], points[:n, 1], "bo-")
# plt.plot(points[n:,0], points[n:,1], "rs-")
def make_prediction_grid(predictors, outcomes, limits, h, k):
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs,ys)
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i, x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

# def plot_prediction_grid (xx, yy, prediction_grid, filename):
#     """ Plot KNN predictions for every point on the grid."""
#     from matplotlib.colors import ListedColormap
#     background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
#     observation_colormap = ListedColormap (["red","blue","green"])
#     plt.figure(figsize =(10,10))
#     plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
#     plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
#     plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
#     plt.xticks(()); plt.yticks(())
#     plt.xlim (np.min(xx), np.max(xx))
#     plt.ylim (np.min(yy), np.max(yy))
#     plt.savefig(filename)
(predictors, outcomes) = generate_synth_data()
# k = 5; filename = "knn_synth_5.pdf"; limits = (-3,4,-3,3); h = 0.1
# (xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
# plot_prediction_grid(xx, yy, prediction_grid, filename)
# print(plt.show())
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(predictors, outcomes)
# sk_predictors = knn.predict(predictors)
# print(sk_predictors[0:10])
# my_predictions = np.array([knn_predict(p,predictors, outcomes, 5) for p in predictors])
# print(my_predictions == sk_predictors)