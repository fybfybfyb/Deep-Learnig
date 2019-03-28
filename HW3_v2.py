import numpy as np
import h5py
#from sklearn.metrics import accuracy_score

np.random.seed(42)

with h5py.File('mnist_traindata.hdf5', 'r') as hf:
    xdata = hf['xdata'][:]
    ydata = hf['ydata'][:]
    
arr = []
for i in range(50000):
    arr.append(True)
for i in range(10000):
    arr.append(False)
np.random.shuffle(arr)
train_x = xdata[arr]
train_y = ydata[arr]
val_x = xdata[[not x for x in arr]]
val_y = ydata[[not x for x in arr]]

def ReLU(x):
	return np.maximum(x, 0)

# derivation of relu
def ReLU_deriv(x):
    x[x <= 0.0] = np.random.random_sample()
    x[x > 0.0] = 1.0
    return x

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class NeuralNetwork:
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 layer,
                 learning_rate,
                 epoch,
                 batch_size = 100,
                 activation = 'ReLU'
                ):  
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        
        self.layer = [x for x in layer]
        self.layer.append(no_of_out_nodes)
            
        self.learning_rate = learning_rate 
        
        self.epoch = epoch
        
        self.batch_size = batch_size
        
        self.activation = activation
        if self.activation == 'ReLU':
            self.activation = ReLU
            self.activation_deriv = ReLU_deriv
        elif self.activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        
        self.build_model()
   
    
    def build_model(self):
        self.weights = []
        self.bias = []
        for i in range(len(self.layer)):
            if i == 0:
                self.weights.append(np.random.normal(loc=0.0, scale=1.0, size=(self.layer[i],self.no_of_in_nodes)))
                self.bias.append(np.random.normal(loc=0.0, scale=1.0, size=(self.layer[i])))
            else:
                self.weights.append(np.random.normal(loc=0.0, scale=1.0, size=(self.layer[i],self.layer[i-1])))
                self.bias.append(np.random.normal(loc=0.0, scale=1.0, size=(self.layer[i])))


    def feed_forward(self,input_vector):
        self.a = []
        self.a_deriv = []
        for i in range(len(self.layer) + 1):
            if i == 0:
                self.a.append(input_vector)
            else:
                output = self.weights[i-1] @ self.a[i-1] + self.bias[i-1]
                self.a.append(self.activation(output))
                self.a_deriv.append(self.activation_deriv(output))
        
        return softmax(self.a[len(self.layer)])


    def backprop(self,input_vector, target_vector):
        self.result = []
        for t in range(self.epoch):
            if t >= 20 and t <= 40:
                self.learning_rate = self.learning_rate / 2
            elif t > 40:
                self.learning_rate = self.learning_rate / 2
                
            for k in range(int(50000/self.batch_size)):
                self.gd_w = []
                self.gd_b = []
                for j in range(self.batch_size):
                    self.feed_forward(input_vector[k*self.batch_size + j])
                    self.delta = []
                    for i in reversed(range(len(self.layer))):
                        if i == len(self.layer) - 1:
                            self.delta.append(softmax(self.a[len(self.layer)]) - target_vector[k*self.batch_size + j])
                        else:
                            self.delta.append(self.a_deriv[i] * (self.weights[i+1].T @ self.delta[len(self.layer)-2-i]))
                    
                    for i in range(len(self.layer)):
                        if j == 0:
                            self.gd_w.append((np.array(self.delta[len(self.layer)-1-i], ndmin = 2) * np.array(self.a[i], ndmin = 2).T).T)
                            self.gd_b.append(self.delta[len(self.layer)-1-i])
                        else:
                            self.gd_w[i] += ((np.array(self.delta[len(self.layer)-1-i], ndmin = 2) * np.array(self.a[i], ndmin = 2).T).T)
                            self.gd_b[i] += (self.delta[len(self.layer)-1-i])
                        
                # do update on weights and bias        
                for i in range(len(self.layer)):
                    self.weights[i] = self.weights[i] - self.learning_rate * (self.gd_w[i] / self.batch_size)
                    self.bias[i] = self.bias[i] - self.learning_rate * (self.gd_b[i] / self.batch_size)
        
            
            self.result.append(np.sum(np.array([x.argmax() for x in val_y]) == np.array([x.argmax() for x in [self.feed_forward(y) for y in val_x]])) / 10000)
            #self.result.append(accuracy_score([x.argmax() for x in val_y],[x.argmax() for x in [self.feed_forward(y) for y in val_x]]))
        
        
if __name__ == "__main__":
     simple_network = NeuralNetwork(no_of_in_nodes = 784, 
                                   no_of_out_nodes = 10, 
                                   layer = (200,100),
                                   learning_rate = 0.001,
                                   epoch = 50)
     simple_network.backprop(train_x,train_y)
     print(simple_network.result)
                 
 
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        