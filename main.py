import csv
import numpy

import pandas as pd
import numpy as np
from numpy import linalg as LA
def SVD(S):
    S_transpose = S.transpose()
    STS = np.dot(S_transpose, S)
    eigenvalues, eigenvectors = LA.eig(STS)
    singular_eigenvalues = np.sqrt(eigenvalues)
    VT = eigenvectors.transpose()
    sigma = np.diag(singular_eigenvalues)
    i = 0
    list_u = []
    for v in eigenvectors:
        u = np.dot(S, v)
        u /= singular_eigenvalues[i]
        list_u.append(u)
    U = np.matrix(list_u, dtype=np.float64)
    U = U.transpose()
    return U,sigma,VT

def SVDpp(S,p):
    U,sigma,VT=SVD(S)
    U_p=U[:,:p]
    sigma_p=sigma[:p,:p]
    VT_p=VT[:p,:]
    return U_p,sigma_p,VT_p


rating=pd.read_csv('./ml-latest-small/ratings.csv')

links=pd.read_csv('./ml-latest-small/links.csv')
tags=pd.read_csv('./ml-latest-small/tags.csv')
movies=pd.read_csv('./ml-latest-small/movies.csv')
rating.drop(columns=['timestamp'])
S=rating.to_numpy()
