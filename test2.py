import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from prini import prin
np.random.seed(123)

def main():
    # Parameters
    n = 40
    d = 10
    s = 15
    m = 10 # iterations
    ns = 2


    print("Given n vectors in d-dimensional Euclidean space where")
    print("n",n)
    print("d",d)
    print("Suppose we have an O(d)-equivariant function f(x_1,...,x_n)")
    print("which outputs a d x d matrix A")
    print("Assume that A = sum_{i j} alpha_{i j} where only s alpha_{i j} nonzero")
    print("s",s)
    print("Suppose we are given the value of the function f for ns random vectors")
    print("ns",ns)
    print("this results in a system of size",(ns*d**2,n**2))
    print("We try the solve this using HTP")

    X = np.random.randn(ns*d,n)
    A = np.zeros((ns*d**2,n**2))

    #print("A.shape",A.shape)

    k = 0
    for i in range(n):
        for j in range(n):
            for itr in range(ns):
                I0 = np.arange(d*itr,(itr+1)*d)
                I1 = np.arange(d**2*itr,(itr+1)*d**2)
                A[I1,k] = (np.outer(X[I0,i],X[I0,j])).flatten()
            k = k +1
    A = A/np.sqrt(A.shape[0])



    idx = np.random.choice(range(n**2), size=s, replace=False)
    idx = np.sort(idx)


    alphas = np.random.randn(s,1)
    b = A[:,idx] @ alphas
    x = np.zeros((n**2,1))
    x[idx] = alphas

    b_check = A @ x


    idx0 = np.random.choice(range(n**2), size=s, replace=False)
    x0 = np.zeros((n**2,1))
    x0[idx0,0] = np.random.randn(s)

    err = np.zeros(m)
    for i in range(m):
        x1 = x0 + A.T @ (b - A @ x0)
        idx1 = np.sort(np.argsort(np.abs(x1).flatten())[-s:])
        tmp = np.zeros((n**2,1))
        tmp[idx1] = np.linalg.lstsq(A[:,idx1],b,rcond=-1)[0]
        x0 = tmp
        err[i] = np.linalg.norm(x0 - x)/np.linalg.norm(x)

    print("idx1",idx1)
    print("idx",idx)
    print("err",err)

    plt.figure()
    plt.plot(err)
    plt.show()
