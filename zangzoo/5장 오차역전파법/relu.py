import numpy as np

class Relu:
    def __init__(self):
        self.mask=None
        
    def forward(self,x):
        # mask라는 인스턴스 변수를 가짐 - T/F로 구성된 넘파이 배열
        # 순전파의 입력 x의 원소값이 0 이하인 인덱스는 True, 0보다 큰 원소는 False로 유지
        # 입력 x 값이 0 이하인 위치를 기억하기 위함
        self.mask = (x<=0)
        # x = np.array([[1, -2], [3, -4]])이면
        # self.mask = np.array([[False, True], [False, True]]) 이렇게 됨
        out = x.copy()
        # self.mask가 True인 위치에 해당하는 값들을 모두 0으로 변경
        # out = [[1, 0], [3, 0]] 으로 변경됨
        out[self.mask]=0
        
        return out
    
    def backward(self,dout):
        # dout에서 True 위치에 해당하는 gradient 값을 0으로 바꿈
        # 만일 dout = np.array([[0.1, 0.2], [0.3, 0.4]])이고,
        # 순전파 값이 위 예시와 같다면,
        # dout = [[0.1, 0.0], [0.3, 0.0]]가 됨 
        # 즉, 순전파에서 음수였던 위치를 0으로 만듦
        dout[self.mask] = 0
        dx=dout
        
        return dx
    
x = np.array([[1, -2], [3, -4]])
