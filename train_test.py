
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=5)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)


reg = LinearRegression(lr=0.009,n_iterations=10000)
reg.fit(X_train,y_train)
predict = reg.predict(X_test)

def mse(predict,y_test):
    return np.mean((predict-y_test)**2)

mse = mse(predict,y_test)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('BuPu_r')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_test,y_test,color=cmap(0.9),s=10)
m2 = plt.scatter(X_train,y_train,color=cmap(0.5),s=10)
plt.plot(X,y_pred_line,color='black', linewidth=2)
plt.show()
