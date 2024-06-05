'''
Task 3.2 Overfitting, underfitting:
In this exercise, we will study the influence of the number of data points and the degree of the parameters
on the model. In order to resolve this exercise, you can use Python, C++, Matlab or other languages.
Consider the following function:
y(x) = cos(2x)
a) Generate a data set D with 30 random data points using the given function. Note that the data set
shouldn't be optimal due to a noise (v~N(u, sigma^2)). Consider reasonable range for x and v(neu).
b) Create polynomials of degree 1, 4 and 15 that will model the data and train them with the data set.
Plot the functions separately, and the generated data set.
c) Which function gives the best solution? What's the problem with the other functions? (keep the
number of data points at 30).
d)What is a possible solution to avoid issues encountered at question c)?
'''

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures



#question a) 
#number of data points
number_samples = 100

#use of the function cos(2pi) in order to create the data
def real_function(X_fct):
    return np.cos(2*np.pi * X_fct)

#X = np.sort(np.random.rand(number_samples))
X = np.linspace(0, 1, number_samples)                       ### 100 sample numbers evenly spaced between 0 and 1
Y = real_function(X) + np.random.randn(number_samples) * 0.1        ### random data + added noise with gaussian distribution






#question b) 
degree = [1,4,15]


for k in range (0,len(degree)):
    #creation of the ML models - here we use the method Linear regression with the polynomial as the phi (chapter 4)
    polynomial_features = PolynomialFeatures(degree=degree[k], include_bias=False)
    linear_regression = LinearRegression()
    #learning process
    pipeline = Pipeline(
            [
                ("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression),
            ]
        )
    pipeline.fit(X[:, np.newaxis], Y)

    #cross-validation (see chapter 12)
    scores = cross_val_score(
            pipeline, X[:, np.newaxis], Y, scoring="neg_mean_squared_error", cv=10
        )




    #plot the functions 
    plt.figure(1)
    if k == 0: 
        plt.subplot(2,2,1)
        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model created")
        plt.plot(X_test, real_function(X_test), label="True function cos(2pi)")
        plt.scatter(X, Y, edgecolor="b", s=20, label="Samples")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.xlim((0, 1))
        plt.ylim((-1.5, 1.5))
        title = "polynomial of degree " + str(degree[k])
        plt.title(title)
        plt.legend()
        
    elif k == 1: 
        plt.subplot(2,2,2)
        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model created")
        plt.plot(X_test, real_function(X_test), label="True function cos(2pi)")
        plt.scatter(X, Y, edgecolor="b", s=20, label="Samples")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.xlim((0, 1))
        plt.ylim((-1.5, 1.5))
        title = "polynomial of degree " + str(degree[k])
        plt.title(title)
        plt.legend()
    elif k == 2: 
        plt.subplot(2,2,3)
        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model created")
        plt.plot(X_test, real_function(X_test), label="True function cos(2pi)")
        plt.scatter(X, Y, edgecolor="b", s=20, label="Samples")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.xlim((0, 1))
        plt.ylim((-1.5, 1.5))
        title = "polynomial of degree " + str(degree[k])
        plt.title(title)
        plt.legend()

    
plt.show()

