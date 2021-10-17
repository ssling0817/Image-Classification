from .layer import *

class Model(object):
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost
        self.pred = None
    def forward(self,x):
        for layer in self.layers:
            #print("forward",type(layer))
            x = layer.forward(x)
        return x

    def loss(self,x,y):
        return self.cost.forward(self.forward(x),y)

    def backward(self):
        grad = self.cost.backward()
        for i in range(len(self.layers)-1,-1,-1):
            #print(type(self.layers[i]))
            if(hasattr(self.layers[i], 'backward')):
                #print("backward",type(self.layers[i]))
                grad = self.layers[i].backward(grad)
        