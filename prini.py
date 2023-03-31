import sys
from numpy import array as array
import numpy as np


def prini():
    fsock = open("out.log", "w")
    fsock.close()


def prin(*args):
    
    if len(args)==1:
        if type(args[0]) == str:
            prin_helper(args[0],0,np.float64,0)
            return
        else:
            msg = "ans"
            x = args[0]
        
    if len(args)==2:
        msg = args[0]
        x = args[1]
   
    float_format = " {:12.5E}"
    float_per_line = 6

    int_format = " {:7d}"
    int_per_line = 10


    complex_format = " {0.real:12.5E}{0.imag:+12.5E}i" 
    complex_per_line = 3
    


    if np.all(x == np.real(x)):
        x = np.real(x)
        if np.all(np.floor(x) == np.float64(x)):
            x = np.array(x).astype(int)
            prin_helper(msg,x,int_format,int_per_line)
        else:
            prin_helper(msg,x,float_format,float_per_line)
    else:
        prin_helper(msg,x,complex_format,complex_per_line)

def prin_helper(msg, x,num_format,per_line):

    x = array(x)
    sz = x.shape
    if len(sz)==0:
        m = 1
        n = 0
    if len(sz)==1:
        m = 1
        n = sz[0]
    if len(sz) >= 2:
        m = sz[0]
        n = sz[1]

    prin_helper2(msg, x, m, n,sz,num_format,per_line)
    saveout = sys.stdout
    fsock = open("out.log", "a")
    sys.stdout = fsock
    prin_helper2(msg, x, m, n,sz,num_format,per_line)
    sys.stdout = saveout
    fsock.close()
    return


def prin_helper2(msg, x, m, n,sz,num_format,per_line):
    #if len(sz) > 0:
    #    sys.stdout.write(" ")
    #    sys.stdout.write(msg)
    #    sys.stdout.write(".shape = ")
    #    sys.stdout.write(str(sz))
    #    sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.write(" ")
    sys.stdout.write(msg)
    if per_line == 0:
        sys.stdout.write("\n")
        return
        
    sys.stdout.write(" = ")
    #sys.stdout.write("{}".format(x.dtype))
    sys.stdout.write("\n")
    n = np.maximum(n,1)
    x = np.reshape(x, (m, n))
    for i in range(m):
        for j in range(n):
            if j % per_line == 0:
                sys.stdout.write("\n    ")
            sys.stdout.write(num_format.format(x[i, j]))
        if n > 6 or i==m-1:
            sys.stdout.write("\n")
    return

# Always Call
prini()
if __name__ == "__main__":
    x = np.random.randn(6, 6)
    prin("x", x)
    x = np.random.randn(12, 12)
    prin("x", x)
