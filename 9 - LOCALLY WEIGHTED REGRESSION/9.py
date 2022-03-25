import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point,xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m))) # eye - identity matrix
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights


def localWeight(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return W


def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred


def graphPlot(X,ypred):

    #Argsort sorts the indices in ascending order of the values and returns the indices
    sortindex = X[:,1].argsort(0)
    xsort = X[sortindex][:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #Plot clusters
    ax.scatter(bill,tip, color='green')

    #Plot the line
    ax.plot(xsort[:,1],ypred[sortindex], color = 'red', linewidth=5)
    plt.xlabel('Total bill')
    plt.ylabel('Tip')
    plt.show();
    
    
    
# load data points
data = pd.read_csv('program9_dataset.csv')

# We use only Bill amount and Tips data
bill = np.array(data.total_bill) 
tip = np.array(data.tip)

# .mat will convert nd array to 2D array
mbill = np.mat(bill)
mtip = np.mat(tip)

#Getting number of records as m
m= len(bill)

#Creating a identity matrix of order m
one = np.mat(np.ones(m))

#Combining the Identity matrix with Bill matrix and craeating a matrix of mx2
X = np.hstack((one.T,mbill.T)) # 244 rows, 2 cols


# increase k to get smooth curves
ypred = localWeightRegression(X,mtip,3)
graphPlot(X,ypred)
