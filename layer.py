import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
## by yourself .Finish your own NN framework

class FullyConnected(_Layer):
    def __init__(self,n_in,n_out):
        self.weights = np.random.randn(n_in,n_out) * np.sqrt(2/n_in)
        self.biases = np.zeros(n_out)

    def forward(self, x):
        self.old_x = x
        return np.dot(x,self.weights) + self.biases

    def backward(self,grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = (np.matmul(self.old_x[:,:,None],grad[:,None,:])).mean(axis=0)
        return np.dot(grad,self.weights.transpose())
## by yourself .Finish your own NN framework
#00 - sigmoid
class ACTIVITY1(_Layer):
    def __init__(self):
        pass

    def forward(self, x):
        self.old_y = np.exp(x) / (1. + np.exp(x))
        return self.old_y


    def backward(self, grad):
        return self.old_y * (1. - self.old_y) * grad
#00 - relu
class Relu(_Layer):
    def __init__(self):
        self.old_x=None;
        pass

    def forward(self, x):     
        self.old_x = np.copy(x)
        return np.clip(x,0,None)


    def backward(self, grad):
        return np.where(self.old_x>0,grad,0)
class Batch_norm(object):
    def __init__(self,name):
        self.X=None
        weight_scale=0.01
        self.gamma=1#np.random.normal(shape=shape, loc=1, scale=weight_scale)
        self.beta = 0#np.random.normal(shape=shape, scale=weight_scale)
        self.momentum = 0.9,
        self.eps = 1e-5,
        self.is_training = True
        self.debug = False
        self.scope_name=name
    def forward(self, X):    
        # mini-batch mean
        self.X=X
        mean = np.mean(X, axis=0)
        # mini-batch variance
        variance = np.mean((X - mean) ** 2, axis=0)
        # normalize
        if self.is_training:
        # while training, we normalize the data using its mean and variance
            X_hat = (X - mean) * 1.0 / np.sqrt(variance + self.eps)
        else:
            # while testing, we normalize the data using the pre-computed mean and variance
            X_hat = (X - _BN_MOVING_MEANS[self.scope_name]) *1.0 / np.sqrt(_BN_MOVING_VARS[self.scope_name] + eps)
            # scale and shift
        out = self.gamma * X_hat + self.beta
        return out
'''
class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        

        #Softmax
        max_value = np.max(input, axis=1)
        max_value = np.repeat(np.expand_dims(max_value, 1), input.shape[1], axis=1)
        offset_values = input - max_value
        exponentiated = np.exp(offset_values)
        sum_exp = np.repeat(np.expand_dims(np.sum(exponentiated, axis=1), 1), input.shape[1], axis=1)
        predict = exponentiated/sum_exp
       

        #Average loss
        x=input
        y=target
        #print(predict.shape,predict)
        #print(target.shape,target)
        N = predict.shape[0]
        your_loss = -np.sum(target*np.log(predict+1e-9))/N
        #your_loss = np.linalg.norm(0.5 * (target - predict))

        return predict, your_loss

    def backward(self):
        max_value = np.max(input, axis=1)
        max_value = np.repeat(np.expand_dims(max_value, 1), input.shape[1], axis=1)
        offset_values = input - max_value
        exponentiated = np.exp(offset_values)
        sum_exp = np.repeat(np.expand_dims(np.sum(exponentiated, axis=1), 1), input.shape[1], axis=1)
        return exponentiated/sum_exp*(1-exponentiated/sum_exp)
     ##   input_grad 

     ##   return input_grad
'''
class Softmax():
    def forward(self,x):
            self.old_y = np.exp(x) / np.exp(x).sum(axis=1) [:,None]
            return self.old_y

    def backward(self,grad):
            return self.old_y * (grad -(grad * self.old_y).sum(axis=1)[:,None])
class CrossEntropy():
    def forward(self,x,y):
            self.old_x = x.clip(min=1e-8,max=None)
            self.old_y = y
            return x,-np.sum(y*np.log(x+1e-9))/x.shape[0]
            #return (np.where(y==1,-np.log(self.old_x), 0)).sum(axis=1)

    def backward(self):
            return np.where(self.old_y==1,-1/self.old_x, 0)
    