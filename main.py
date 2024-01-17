import csv
import numpy

import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm


def SVD(S):
    width,height=S.shape
    S_transpose = S.transpose()
    STS = np.dot(S,S_transpose)
    eigenvalues, eigenvectors = LA.eig(STS)
    # print("hello")

    singular_eigenvalues = np.sqrt(eigenvalues)
    # sigma_p=np.zeros((width,height),eigenvalues.dtype)


    sigma = np.diag(singular_eigenvalues)
    sigma =np.pad(sigma,((0,width-sigma.shape[0]),(0,height-sigma.shape[1])),constant_values=(0,0))
    # print(sigma.shape)
    i = 0
    list_u = []
    # print(eigenvectors.shape)
    # print(singular_eigenvalues)


    # number_zero_eigenvector=S.shape[1]-eigenvectors.shape[1]
    # zero_eigenvector=np.zeros((eigenvectors.shape[0],number_zero_eigenvector))
    # VT = np.hstack((eigenvectors,zero_eigenvector))
    # print(height)
    # print(eigenvectors.shape[0])
    # print(eigenvectors.shape[1])
    # print(height-eigenvectors.shape[0])

    VT=np.pad(eigenvectors,((0,height-eigenvectors.shape[0]),(0,height-eigenvectors.shape[1])),constant_values=(0,0))
    # print(VT.shape)
    for v in eigenvectors:

        # print(S.shape)
        # print(v.shape)
        # print(v.shape)
        vT=np.pad(v,(0,height-v.shape[0]),constant_values=(0,0))
        # print(VT)
        u = np.dot(S,vT)
        # if(singular_eigenvalues[i]==0):
            # print('fuck')
        u = u/singular_eigenvalues[i]
        list_u.append(u)
        # print(u.shape)
        i+=1
    U = np.matrix(list_u, dtype=np.float64)
    # U = U.transpose()

    return U,sigma,VT

def SVDpp(S,p):
    U,sigma,VT=SVD(S)
    U_p=U[:,:p]
    sigma_p=sigma[:p,:p]
    VT_p=VT[:p,:]
    return U_p,sigma_p,VT_p



# def cosine_similarity(A,B):
#     return  np.dot(A,B)/(norm(A)*norm(B))



def prediction(user_id,S):
    U, sigma, VT = SVD(S)
    cosine_similarity = (U[user_id].dot(sigma)).dot(VT) / (norm(VT) * norm(U[user_id].dot(sigma)))
    return cosine_similarity





rating=pd.read_csv('./ml-latest-small/ratings.csv')

links=pd.read_csv('./ml-latest-small/links.csv')
tags=pd.read_csv('./ml-latest-small/tags.csv')
movies=pd.read_csv('./ml-latest-small/movies.csv')
# print(type(movies.loc[1][0]))
# print(movies.shape[0])

NUMBER_OF_RATED_MOVIES=int(movies.shape[0])
NUMBER_OF_USERS=int(rating.max(axis='rows')[0])
# print(np.where(movies['movieId']==1)[0][0])
S = np.zeros((NUMBER_OF_USERS,NUMBER_OF_RATED_MOVIES))
# print(S.shape)
for i in range(0,np.shape(rating)[0]):
    S[int(rating.loc[i][0])-1][np.where(movies['movieId']==int(rating.loc[i][1]))[0][0]]=rating.loc[i][2]
#
# print(S.shape)
# U,sigma,VT=SVD(S)
# print(U.shape)
# print(sigma.shape)
# print(VT.shape)
# print(VT)
# S=rating.to_numpy()
#
#
print("please enter user_id :")
a=int(input())

predict=prediction(a-1,S)
predict=np.ravel(predict)
ind=np.argpartition(predict,-10)[-10:]
print(ind[np.argsort(predict[ind])])
for i in ind[np.argsort(predict[ind])]:
    print(movies.loc[i])
# indices=indices.flatten()
# print(indices.shape)
# print(indices)
# for i in range(0,indices.shape[1]-1):
#     print(indices[0,i])
#     print(i)
# indice=np.argsort(indices)[:-10]
# print(indice)
# for i in indice:
#
#     print(i)
    # print(movies.loc[i])



