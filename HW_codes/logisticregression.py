import numpy as np 
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size = n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc = 0, scale = 1, size = n)
# plt.figure()
# plt.plot(x, y, "o", ms = 5)
xx = np.array([0, 10])
# plt.plot(xx, beta_0+ beta_1 * xx)
# plt.xlabel("x")
# plt.ylabel("y")
def compute_rss(y_estimate, y): 
  return sum(np.power(y-y_estimate, 2)) 
def estimate_y(x, b_0, b_1): 
  return b_0 + b_1 * x 
rss = []
slopes = np.arange(-10, 15, 0.001)
for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope * x)**2))

ind_min = np.argmin(rss)
# plt.figure()
# plt.plot(slopes, rss)
# plt.xlabel("Slope")
# plt.ylabel("RSS")
mod = sm.OLS(y, x)
est = mod.fit()
X = sm.add_constant(x)
mod = sm.OLS(y, X)
est = mod.fit()
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10*ss.uniform.rvs(size = n)
x_2 = 10*ss.uniform.rvs(size = n)
y = beta_0 + beta_1 * x_1 + beta_2 * x_2+ ss.norm.rvs(loc = 0, scale = 1, size = n)
X = np.stack([x_1, x_2], axis = 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1],y, c= y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.5, random_state = 1)
lm = LinearRegression(fit_intercept = True)
lm.fit(X_train, y_train)
lm.score(X_test, y_test)
def gen_data(n, h, sd1, sd2):
  x1 = ss.norm.rvs(-h, sd1, n)
  y1 = ss.norm.rvs(0, sd1, n)

  x2 = ss.norm.rvs(h, sd2, n)
  y2 = ss.norm.rvs(0, sd2, n)
  return (x1, y1, x2, y2)
(x1, y1, x2, y2) = gen_data(1000, 1.5, 1, 1.5)
def plot_data(x1, y1, x2, y2):
  plt.figure()
  plt.plot(x1, y1, "o", ms = 2)
  plt.plot(x2, y2, "o", ms = 2)
  plt.xlabel("$X_1$")
  plt.ylabel("$X_2$")
plot_data(x1, y1, x2, y2)
def prob_to_odds(p): 
    if p <= 0 or p >= 1: 
        print("Probabilities must be between 0 and 1.") 
    return p / (1-p) 
clf = LogisticRegression()
n = 1000
X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
y = np.hstack((np.repeat(1, n), np.repeat(2, n)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, random_state = 1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
def plot_probs(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    Z = probs[:,class_no]
    Z = Z.reshape(xx1.shape)
    CS = ax.contourf(xx1, xx2, Z)
    cbar = plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2");