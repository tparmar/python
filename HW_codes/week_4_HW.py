import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import warnings

df = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@movie_data.csv", index_col=0)
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)
regression_target = 'revenue'
classification_target = 'profitable'
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
list_genres = df.genres.apply(lambda x: x.split(","))
genres = []
for row in list_genres:
    row = [genre.strip() for genre in row]
    for genre in row:
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    df[genre] = df['genres'].str.contains(genre).astype(int)

continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15, \
       color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
# show the plot.

# determine the skew.
df[outcomes_and_continuous_covariates].skew()

# temp2 = ["budget", "popularity", "runtime", "vote_count", "revenue"]
# for i in temp2:
#     for j in range(len(df[i])):
#         df[i] = np.log10(1+df[i][j])
for covariate in ['budget', 'popularity', 'runtime', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))
df.to_csv("movies_clean.csv")
warnings.filterwarnings("ignore")

# EDIT THIS CODE TO LOAD THE SAVED DF FROM THE LAST HOMEWORK
df = pd.read_csv('movies_clean.csv')

regression_target = 'revenue'
classification_target = 'profitable'
all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy', 
                  'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance', 
                  'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)

def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)
# Determine the cross-validated correlation for linear and random forest models.
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
# Plot Results
# plt.axes().set_aspect('equal', 'box')
# plt.scatter(linear_regression_scores, forest_regression_scores)
# plt.plot((0, 1), (0, 1), 'k-')

# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel("Linear Regression Score")
# plt.ylabel("Forest Regression Score")

# Show the plot.
# plt.show()
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

# plt.axes().set_aspect('equal', 'box')
# plt.scatter(logistic_regression_scores, forest_classification_scores)
# plt.plot((0, 1), (0, 1), 'k-')

# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel("Linear Classification Score")
# plt.ylabel("Forest Classification Score")

# Show the plot.
# plt.show()

positive_revenue_df = df.loc[df["revenue"] > 0]

# Replace the dataframe in the following code, and run.
regression_outcome = positive_revenue_df[regression_target]
classification_outcome = positive_revenue_df[classification_target]
covariates = positive_revenue_df[all_covariates]

# Reinstantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

# Determine the cross-validated correlation for linear and random forest models.
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv = 10, scoring = correlation)
forest_regression_scores =cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
# Plot Results
# plt.axes().set_aspect('equal', 'box')
# plt.scatter(linear_regression_scores, forest_regression_scores)
# plt.plot((0, 1), (0, 1), 'k-')

# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel("Linear Regression Score")
# plt.ylabel("Forest Regression Score")

# Show the plot.
# plt.show()
# Print the importance of each covariate in the random forest regression.
forest_regression.fit(covariates, regression_outcome)
sorted(list(zip(all_covariates, forest_regression.feature_importances_)), key=lambda tup: tup[1])

# Determine the cross-validated accuracy for logistic and random forest models.
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)
# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

# Show the plot.
plt.show()
# Print the importance of each covariate in the random forest classification.
forest_classifier.fit(covariates, classification_outcome)
print(sorted(list(zip(all_covariates, forest_classifier.feature_importances_)), key=lambda tup: tup[1]))
