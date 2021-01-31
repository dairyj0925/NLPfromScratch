import numpy as np
from common.functions import softmax, cross_entropy_error

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x 
        return out
    
    def backward(self, dout):
        W, =  self.params
        dx = np.dot(dout, W.T)
        dw = np.dot(self.x.T, dout)
        self.grads[0][...] = dw

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self,w,b):
        self.params = [w,b]
        self.grads = [np.zeros_like(w),np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        w,b = self.params
        out = np.dot(x, w) + b
        self.x = x 
        return out
    
    def backward(self, dout):
        w,b = self.params
        dx = np.dot(dout, w.T)
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        # y^2 - sum(y^2) * y
     
        dx = self.out * dout
        
        sumdx = np.sum(dx, axis=1, keepdims=True)
     
        dx -= self.out * sumdx
       
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 在监督标签为one-hot向量的情况下，转换为正确解标签的索引
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


if __name__ == "__main__":
    '''
    Sigmoid
    '''
    # x = np.random.rand(1,2)
    # w = np.random.rand(2,4)

    # layer = Sigmoid()
    # out = layer.forward(x)
    # print(x)
    # print(out)
    
    # dx = layer.backward(out)
    # print(dx)

    '''
    Affine
    '''
    # x = np.random.rand(1,2)
    # w = np.random.rand(2,4)
    # b = np.random.rand(1,4)
    # a = Affine(w,b)

    # out = a.forward(x)
    # print(out)
    # dx = a.backward(out)
    # print(dx)

    '''
    Softmax
    '''
    # x = np.random.rand(1,2)
    # t = [[1,0]]
    # l = Softmax()
    # out = l.forward(x)
    
    # print(out)

    # dx = l.backward(out)
    # print(dx)