import numpy as np, random, scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.preprocessing
import sklearn.decomposition
from sklearn.neighbors import KNeighborsClassifier
def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]
data = pd.read_csv("data_case_study_3.txt")
numeric_data = data.rename(columns={"color": "is_red"})
for color in numeric_data["is_red"]:
    if color == "red":
        numeric_data["is_red"].replace({"red": 1}, inplace=True)
    elif color == "white":
        numeric_data["is_red"].replace({"white":0}, inplace=True)

scaled_data = sklearn.preprocessing.scale(numeric_data)
numeric_data = pd.DataFrame(scaled_data, columns = numeric_data.columns)


pca = sklearn.decomposition.PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data)


observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")


np.random.seed(1) 
x = np.random.randint(0, 2, 1000)
y = np.random.randint(0 ,2, 1000)
def accuracy(predictions, outcomes):
    x = predictions == outcomes
    percent = len(x[True])/len(x)
    return percent
print(accuracy(x,y))

def accuracy(predictions, outcomes):

       return 100*np.mean(predictions == outcomes)
data["is_red"] = (data["color"] == "red").astype(int)
numeric_data = data.drop(["color", "high_quality", "quality"], axis=1)

numeric_data.groupby('is_red').count()
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
random.seed(123)
n_rows = data.shape[0]
selection = random.sample(range(n_rows), 10)
predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])

my_predictions = my_predictions = np.array( [ knn_predict(p, predictors[training_indices,:], outcomes[training_indices], 5) for p in predictors[selection] ])  
percentage = accuracy(my_predictions, data.high_quality.iloc[selection])
print(percentage)
