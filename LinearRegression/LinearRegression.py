from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from scipy import stats
import math

X, Y = datasets.make_regression(n_samples=400, n_features=1, n_targets=1, noise=25)

# print(stats.shapiro(X))
# print(stats.shapiro(Y))
Xmean = np.mean(X)
Ymean = np.mean(Y)

covXY = 0
varX = 0
for x, y in zip(X.flatten(), Y):
    covXY += (x - Xmean) * (y - Ymean)
    varX += math.pow(x - Xmean, 2)

beta = covXY / varX
alpha = Ymean - Xmean * beta

predict = beta * X + alpha

plt.plot(X, predict, c="red")
plt.scatter(X, Y)
plt.show()
