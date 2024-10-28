import numpy as np

X_dot_W=np.array([[0,0,0],[10,10,10]])
B=np.array([1,2,3])

print(X_dot_W+B)
#[[ 1  2  3]
#[11 12 13]]


dY=np.array([[1,2,3],[4,5,6]])
dB=np.sum(dY, axis=0)

print(dB)
#[5 7 9]

