import numpy as np

def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error_1(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

def cross_entropy_error(y,t):
    if y.dim==1:
        t=t.reshape(1, t.shape)
        y=y.reshape(1, y.size)
        
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))/batch_size

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h)) / (2*h)

def function_2(x):
    return x[0]**2+x[1]**2

def f_tmp1(x0):
    return x0*x0 + 4.0**2.0
    
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val=x[idx]
        # f(x+h) 계산
        x[idx]=tmp_val+h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx]=tmp_val-h
        fxh2=f(x)
        
        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val # 값 복원
        
    return grad

#print(numerical_gradient(function_2,np.array([3.0,4.0])))

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x=init_x
    
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad  
    return x
