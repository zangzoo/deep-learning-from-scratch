import numpy as np

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    print(y)
    return y

a = np.array([1010,1000,900])
softmax(a)

# 오버플로 해결
def softmax_solve_problem(a):
    c=np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    print(y)
    return y
softmax_solve_problem(a)
