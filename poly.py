# Machine Learning 2
# Tinus Strydom
# Program implementing polimonial regression

#Imports packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#using dataframe read csv file for data 
datas = pd.read_csv('external_debt.csv')

#Assign to X the column of rank
X = datas.iloc[:, 1:2].values
#Assign to y the column of External debt
y = datas.iloc[:, 2].values

#Spilt data in training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Linear regression
#assign to lin_reg the linear regression method
lin_reg = LinearRegression()
#train the data with fit
lin_reg.fit(X,y)

#define the linear visualization
def linearVisualize():
    #scatter the values of X,y with 
    plt.scatter(X,y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    return

#Polymonial regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

#define poly visualization
def polyVisualize():
    #scatter the values of X,y with color yellow
    plt.scatter(X,y, color='yellow')
    #
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='purple')
    return

#Main Method
def main():
    linearVisualize()
    polyVisualize()
    plt.title('Countries By External Debt')
    plt.xlabel('Rank')
    plt.ylabel('External Debt')
    plt.show()


if __name__ == "__main__":
    main()
