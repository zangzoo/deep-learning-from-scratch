import numpy as np
import matplotlib.pylab as plt

#계단함수
def step_function(x):
    return np.array(x>0, dtype=np.int)

x=np.arange(-5.0, 5.0, 0.1) 
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

#시그모이드 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))
x=np.arange(-5.0, 5.0, 0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

#ReLU 함수
def relu(x):
    return np.maximum(0,x)
x=np.arange(-5.0, 5.0, 0.1)
y=relu(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

