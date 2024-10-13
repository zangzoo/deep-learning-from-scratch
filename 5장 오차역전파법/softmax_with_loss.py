import numpy as np

def softmax(a):
    c=np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    print(y)
    return y

def cross_entropy_error(y,t):
    if y.dim==1:
        t=t.reshape(1, t.shape)
        y=y.reshape(1, y.size)
        
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))/batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None # 손실
        self.y = None # 소프트맥스의 출력
        self.t = None # 정답 레이블 (원핫벡터)
        
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        
        return dx