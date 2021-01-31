import sys
sys.path.append("..")
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I,H,O = input_size, hidden_size, output_size

        w1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        w2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params , self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score  = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def back



