import csv
import numpy

import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm


def SVD(S):
    S_transpose = S.transpose()
    STS = np.dot(S,S_transpose)
    eigenvalues, eigenvectors = LA.eig(STS)
    singular_eigenvalues = np.sqrt(eigenvalues)

    sigma = np.diag(singular_eigenvalues)
    i = 0
    list_u = []
    # print(eigenvectors.shape)
    # print(singular_eigenvalues)
    number_zero_eigenvector=S.shape[1]-eigenvectors.shape[1]
    zero_eigenvector=np.zeros((eigenvectors.shape[0],number_zero_eigenvector))
    VT = np.hstack((eigenvectors,zero_eigenvector))
    for v in VT:
        # print(v.shape)
        # print(S.shape)
        # print(v.shape)
        u = np.dot(S,v)
        if(singular_eigenvalues[i]==0):
            print('fuck')
        u = u/singular_eigenvalues[i]
        list_u.append(u)
        # print(u.shape)
        i+=1
    U = np.matrix(list_u, dtype=np.float64)
    U = U.transpose()

    return U,sigma,VT

def SVDpp(S,p):
    U,sigma,VT=SVD(S)
    U_p=U[:,:p]
    sigma_p=sigma[:p,:p]
    VT_p=VT[:p,:]
    return U_p,sigma_p,VT_p

def cosine_similarity(mat,vec,k):
   cosine= np.dot(mat, vec) / (norm(mat, axis=1) * norm(vec))
   cosine=np.sort(cosine)
   return cosine[:k]

def cosine_similarity(A,B):
    return  np.dot(A,B)/(norm(A)*norm(B))



def prediction(user_id,S,p):
    U, sigma, VT = SVDpp(S,p)





rating=pd.read_csv('./ml-latest-small/ratings.csv')

links=pd.read_csv('./ml-latest-small/links.csv')
tags=pd.read_csv('./ml-latest-small/tags.csv')
movies=pd.read_csv('./ml-latest-small/movies.csv')
# print(type(movies.loc[1][0]))
NUMBER_OF_MOVIES=int(movies.max(axis='rows')[0])
NUMBER_OF_USERS=int(rating.max(axis='rows')[0])

S = np.zeros((NUMBER_OF_USERS,NUMBER_OF_MOVIES))
for i in range(0,np.shape(rating)[0]):
    S[int(rating.loc[i][0])-1][int(rating.loc[i][1])-1]=rating.loc[i][2]

print(S.shape)
U,sigma,VT=SVD(S)
print(U.shape)
print(sigma.shape)
print(VT.shape)
# S=rating.to_numpy()


