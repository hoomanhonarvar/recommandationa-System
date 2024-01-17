

import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm

#creating the SVD form of matrix S
def SVD(S):
    width,height=S.shape
    S_transpose = S.transpose()
    STS = np.dot(S,S_transpose)
    eigenvalues, eigenvectors = LA.eig(STS)
    singular_eigenvalues = np.sqrt(eigenvalues)
    sigma = np.diag(singular_eigenvalues)
    sigma =np.pad(sigma,((0,width-sigma.shape[0]),(0,height-sigma.shape[1])),constant_values=(0,0))
    i = 0
    list_u = []
    VT=np.pad(eigenvectors,((0,height-eigenvectors.shape[0]),(0,height-eigenvectors.shape[1])),constant_values=(0,0))
    for v in eigenvectors:
        vT=np.pad(v,(0,height-v.shape[0]),constant_values=(0,0))
        u = np.dot(S,vT)
        u = u/singular_eigenvalues[i]
        list_u.append(u)
        i+=1
    U = np.matrix(list_u, dtype=np.float64)

    return U,sigma,VT


#calucateing SVD plus with compression
def SVDpp(S,p):
    U,sigma,VT=SVD(S)
    U_p=U[:,:p]
    sigma_p=sigma[:p,:p]
    VT_p=VT[:p,:]
    return U_p,sigma_p,VT_p



#getting the cosine_similarity of prediction of recommendation model
def prediction(user_id,S):
    U, sigma, VT = SVD(S)
    cosine_similarity = (U[user_id].dot(sigma)).dot(VT) / (norm(VT) * norm(U[user_id].dot(sigma)))
    return cosine_similarity





rating=pd.read_csv('./ml-latest-small/ratings.csv')

links=pd.read_csv('./ml-latest-small/links.csv')
tags=pd.read_csv('./ml-latest-small/tags.csv')
movies=pd.read_csv('./ml-latest-small/movies.csv')

NUMBER_OF_RATED_MOVIES=int(movies.shape[0])
NUMBER_OF_USERS=int(rating.max(axis='rows')[0])


S = np.zeros((NUMBER_OF_USERS,NUMBER_OF_RATED_MOVIES))

for i in range(0,np.shape(rating)[0]):
    S[int(rating.loc[i][0])-1][np.where(movies['movieId']==int(rating.loc[i][1]))[0][0]]=rating.loc[i][2]

print("please enter user_id :")
a=int(input())

predict=prediction(a-1,S)
#changing matrix form to array
predict=np.ravel(predict)
#picking index of 10 first biggest elements in array
ind=np.argpartition(predict,-10)[-10:]
print(ind[np.argsort(predict[ind])])
for i in ind[np.argsort(predict[ind])]:
    print(movies.loc[i])


