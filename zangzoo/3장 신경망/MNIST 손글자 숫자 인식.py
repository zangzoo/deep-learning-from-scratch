# 이미 학습된 매개변수 사용 -> 학습과정 생략, 추론 과정만 구현
# 즉, 순전파만 구현

import sys,os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train,t_train), (x_test, t_test)=\
        load_mnist(flatten=True, normalize=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax_solve_problem(a):
    c=np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    print(y)
    return y

def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_solve_problem(a3)

    return y

x,t=get_data()
network=init_network()

batch_size=100 # 배치 크기, 덩어리 크기
accuracy_cnt=0

for i in range(0,len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p=np.argmax(y_batch,axis=1) #확률 가장 높은 인덱스
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])
print('acc:'+str(float(accuracy_cnt)/len(x)))

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)


