from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets


X,Y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=25)

# plt.scatter(X,Y,linewidths=0.1)
# plt.show()

model = LinearRegression()
model.fit(X,Y)

print(X[:200,:].shape)
predict = model.predict(X)

plt.plot(X,predict,c="red")
plt.scatter(X,Y)
plt.show()