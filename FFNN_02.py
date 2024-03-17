#!/usr/bin/env python
# coding: utf-8

# # New Section

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split


# In[2]:


def sigmoid(x):
     z = np.exp(-x)
     return 1/(1+z)

def exp (y) :
	return (np.e)**y

def d_sigmoid(x):
  return (1 - sigmoid(x)) * sigmoid(x)

def tanh(x):
  return np.tanh(x)

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

def relu(x):
  return np.where(np.asarray(x) > 0, x, 0)

def d_relu(x):
    return np.where(x <= 0, 0, 1)

def softmax(x):
    #e_x = np.exp(x)
    #return e_x/e_x.sum()
    # A = np.exp(x)/np.exp(x).sum()
    return exp(x) / np.sum(exp(x))
    # cache = x
    # return A


"""def d_sigmoid(x) :
    return np.multiply(d_sigmoid(x), ( np.ones(x.shape) - d_sigmoid(x) ))

def Reshape (vector):
    return vector.reshape(vector.shape[0],1)

def relu(x) :
    x= x / np.max(x)
    y= np.maximum(0,x)
    #print (y)
    return y

def d_relu (x):
    y= np.zeros(x.shape)
    for i in range (len (x)):
        y[i]=1 if x[i]>=0 else 0
    return y

def tanh (x):
    return (exp(x) - exp (-x))/(exp(x) + exp (-x))

def d_tanh (x) :
    return (np.ones(x.shape) - np.square(tanh(x)))

"""


# In[3]:


#cross-entropy
def cross_entropy_loss(y, y_hat, i):
  return -np.log(y_hat[y[i]][0])



""" def cross_entropy(out):
        loss = -1 *  np.log(out)
        return loss
   epsilon = 1e-9  # Small constant to avoid log(0)
    m = len(y)
    cost = 0
    for i in range(m):
        if y_hat[y[i]][0] <= 0:
            print(f"Error: Invalid probability encountered for index {y[i]}.")
            return None
        cost += -np.log(y_hat[y[i]][0] + epsilon)  # Add epsilon to prevent log(0)
    return cost
     """


# In[4]:


class Layer:

    activationFunc = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid),
        'relu' : (relu, d_relu),
        'softmax' : (softmax, None)
    }

    def __init__(self, inputs, neurons, activation):

        #Xavier initialization
        np.random.seed(33)
        sd = np.sqrt(2 / float(inputs + neurons))
        self.W = np.random.normal(0, sd, size=(neurons, inputs))
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activationFunc.get(activation)
        self.dW = 0
        self.db = 0


# In[5]:


"""def forwardPropagate(self, X_train_batch, weights, biases):

        Returns the neural network given input data, weights, biases.
        Arguments:
                 : X_train_batch - input matrix
                 : Weights  - Weights matrix
                 : biases - Bias vectors

        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        # A - Preactivations
        # H - Activations
        X = X_train_batch
        H = {}
        A = {}
        H["0"] = X
        A["0"] = X
        for l in range(0, num_layers - 2):
            if l == 0:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, X), b)
                H[str(l + 1)] = self.activation(A[str(l + 1)])
            else:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, H[str(l)]), b)
                H[str(l + 1)] = self.activation(A[str(l + 1)])

        # Here the last layer is not activated as it is a regression problem
        W = weights[str(num_layers - 1)]
        b = biases[str(num_layers - 1)]
        A[str(num_layers - 1)] = np.add(np.matmul(W, H[str(num_layers - 2)]), b)
        # Y = softmax(A[-1])
        Y = softmax(A[str(num_layers - 1)])
        H[str(num_layers - 1)] = Y
        return Y, H, A

def forward_propagation(h, layers):
  m = len(layers)

  layers[0].a = np.dot(layers[0].W, h)
  layers[0].h = layers[0].act(layers[0].a)

  for j in range(1, m-1):
        if layers[j].W.shape[0] != layers[j-1].h.shape[0]:
                print(f"Error: Dimension mismatch in layer {j}.")
                return None
        layers[j].a = np.dot(layers[j].W, layers[j-1].h)
        layers[j].h = layers[j].act(layers[j].a)

      if layers[j].W.shape[0] != layers[j-1].h.shape[0]:
            print(f"Error: Dimension mismatch in output layer {j}.")
            return None
      j+=1
      layers[j].a = np.dot(layers[j].W, layers[j-1].h)
      layers[j].h = softmax(layers[j].a)

      return layers[m-1].h

  for j in range(m):
        # Check dimension mismatch before matrix multiplication
        if j == 0:
            if layers[j].W.shape[1] != h.shape[0]:
                print(f"Error: Dimension mismatch in layer {j}.")
                return None
            layers[j].a = np.dot(layers[j].W, h) + layers[j].b
        else:
            if layers[j].W.shape[1] != layers[j-1].h.shape[0]:
                print(f"Error: Dimension mismatch in layer {j}.")
                return None
            layers[j].a = np.dot(layers[j].W, layers[j-1].h) + layers[j].b

        # Apply activation function
        layers[j].h = layers[j].act(layers[j].a)

    # Softmax for output layer
  layers[-1].h = softmax(layers[-1].a)

  return layers[-1].h

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0))  # Subtract max to avoid overflow
    return exp_x / exp_x.sum(axis=0)
    """

def forward_propagation(h, layers):
    m = len(layers)

    layers[0].a = np.dot(layers[0].W, h)
    layers[0].h = layers[0].act(layers[0].a)

    for j in range(1, m-1):
      layers[j].a = np.dot(layers[j].W, layers[j-1].h)
      layers[j].h = layers[j].act(layers[j].a)

    j+=1
    layers[j].a = np.dot(layers[j].W, layers[j-1].h)
    layers[j].h = softmax(layers[j].a)


    return layers[m-1].h


# In[5]:





# In[6]:


def backward_propagation(l, y_hat, layers, inp):

  #one-hot vector
  e_l = np.zeros((y_hat.shape[0], 1))
  e_l[l] = 1

  layers[len(layers)-1].da = -(e_l - y_hat)                 #gradient w.r.t activation of last layer (a_L)

  for j in range(len(layers)-1, 0, -1):

    layers[j].dW += np.dot(layers[j].da, (layers[j-1].h).T)
    layers[j].db += layers[j].da

    layers[j-1].dh = np.dot((layers[j].W).T, layers[j].da)
    layers[j-1].da = np.multiply(layers[j-1].dh, layers[j-1].d_act(layers[j-1].a))

  layers[0].dW += np.dot(layers[0].da, inp.T)
  layers[0].db += layers[0].da

  return layers


# In[7]:


def sgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      cost = 0
      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)
        #cost += cross_entropy_loss(y_train, h, i)

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #stocastic gradient decent
        if (i+1) % batch_size == 0:
          for layer in layers:
            layer.W = layer.W - learning_rate * layer.dW/batch_size
            layer.b = layer.b - learning_rate * layer.db/batch_size

            layer.dW = 0
            layer.db = 0

      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0
      for i in range(len(y_val)):
      #loss += squared_error(y, prediction[:, i].reshape(10,1), i)
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Cost: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers


# In[8]:


def mgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    gamma = 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:
        layer.update_W = 0
        layer.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #momentum gradient decent
        if (i+1) % batch_size == 0:
          for layer in layers:

            layer.update_W = gamma*layer.update_W + learning_rate*layer.dW/batch_size
            layer.update_b = gamma*layer.update_b + learning_rate*layer.dW/batch_size

            layer.W = layer.W - layer.update_W
            layer.b = layer.b - layer.update_b

            layer.dW = 0
            layer.db = 0

            layer.update_W = 0
            layer.update_b = 0


      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0
      for i in range(len(y_val)):
      #loss += squared_error(y, prediction[:, i].reshape(10,1), i)
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Cost: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers


# In[9]:


def nesterov(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    gamma = 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:
        layer.update_W = 0
        layer.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)

        #calculate W_lookaheads
        if (i+1) % batch_size == 0:
          for layer in layers:
            layer.W = layer.W - gamma * layer.update_W
            layer.b = layer.b - gamma * layer.update_b

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #nesterov gradient decent
        if (i+1) % batch_size == 0:
          for layer in layers:

            layer.update_W = gamma*layer.update_W + learning_rate*layer.dW/batch_size
            layer.update_b = gamma*layer.update_b + learning_rate*layer.dW/batch_size

            layer.W = layer.W - layer.update_W
            layer.b = layer.b - layer.update_b

            layer.dW = 0
            layer.db = 0

            layer.update_W = 0
            layer.update_b = 0

      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0
      for i in range(len(y_val)):
      #loss += squared_error(y, prediction[:, i].reshape(10,1), i)
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Cost: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers


# In[10]:


def adam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    epsilon, beta1, beta2 = 1e-8, 0.9, 0.99
    t = 0

    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:
        layer.m_W, layer.m_b, layer.v_W, layer.v_b, layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #adam gradient decent
        if (i+1) % batch_size == 0:
          t+=1

          for layer in layers:

            layer.m_W = beta1 * layer.m_W + (1-beta1)*layer.dW/batch_size
            layer.m_b = beta1 * layer.m_b + (1-beta1)*layer.db/batch_size

            layer.v_W = beta2 * layer.v_W + (1-beta2)*((layer.dW/batch_size))**2
            layer.v_b = beta2 * layer.v_b + (1-beta2)*((layer.db/batch_size))**2

            layer.m_W_hat = layer.m_W/(1-math.pow(beta1, t))
            layer.m_b_hat = layer.m_b/(1-math.pow(beta1, t))

            layer.v_W_hat = layer.v_W/(1-math.pow(beta2, t))
            layer.v_b_hat = layer.v_b/(1-math.pow(beta2, t))

            layer.W = layer.W - (learning_rate/np.sqrt(layer.v_W_hat + epsilon))*layer.m_W_hat
            layer.b = layer.b - (learning_rate/np.sqrt(layer.v_b_hat + epsilon))*layer.m_b_hat

            layer.dW = 0
            layer.db = 0

            layer.m_W, layer.m_b, layer.v_W, layer.v_b, layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0


      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0
      for i in range(len(y_val)):
      #loss += squared_error(y, prediction[:, i].reshape(10,1), i)
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      #wandb.log({"epoch": epoch, "loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Cost: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers


# In[11]:


def optimizor(layers, optimizer, epochs, learning_rate, x_train, y_train, x_val, y_val, batch_size):

  if optimizer == "sgd":
    return sgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "mgd":
    return mgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "nesterov":
    return nesterov(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "adam":
    return adam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  else:
    print("No optimization algorithm named "+optimizer+" found")
    return "Error", "Error"


# In[12]:


def predict(input, y, layers):

  prediction = forward_propagation(input, layers)

  loss = 0
  for i in range(len(y)):
    #loss += squared_error(y, prediction[:, i].reshape(10,1), i)
    loss += cross_entropy_loss(y, prediction[:, i].reshape(10,1), i)

  prediction = prediction.argmax(axis=0)
  accuracy =  np.sum(prediction == y)/y.shape[0]

  return prediction, accuracy, loss/len(y)


# In[13]:


from keras.datasets import fashion_mnist
(x_train_org, y_train_org), (x_test_org, y_test_org) = fashion_mnist.load_data()


# In[14]:


print("x_train shape: ", x_train_org.shape)
print("y_train shape: ", y_train_org.shape)


# In[15]:


x_train_temp = x_train_org.reshape(x_train_org.shape[0], -1)
y_train_temp = y_train_org
x_test = x_test_org.reshape(x_test_org.shape[0], -1)
y_test = y_test_org


# In[16]:


x_train_norm= x_train_org/255.0
x_test_norm= x_test_org/255.0


# In[16]:





# In[17]:


x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.1, random_state=33)
#x_train, x_val, y_train, y_val = train_test_split(x_train_norm, y_train_temp, test_size=0.1, random_state=33)


# In[18]:


print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_val shape: ", x_val.shape)
print("y_val shape: ", y_val.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)


# In[19]:


def model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val):
    #with wandb.init(config=config):
    #config = wandb.config
    #epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val
    layers= [Layer(x_train.shape[1], neurons, activation)]
    for _ in range(0, h_layers-1):
      layers.append(Layer(neurons, neurons, activation))
      layers.append(Layer(neurons, 10, 'softmax'))

      costs, layers = optimizor(layers, optimizer, epochs, learning_rate, x_train, y_train, x_val, y_val, batch_size)

      output_test, accuracy_test, loss_test = predict(x_test.T, y_test, layers)


    print("Test accuracy: ", accuracy_test)
    print("Test loss: ", loss_test)

    return output_test


# In[20]:


"""
def model_train(config=None):
  with wandb.init(config=config):
    config = wandb.config
    #epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val
    layers= [Layer(x_train.shape[1], config.neurons, config.activation)]
    for _ in range(0, h_layers-1):
      layers.append(Layer(config.neurons, config.neurons, config.activation))
      layers.append(Layer(config.neurons, 10, 'softmax'))

      costs, layers = optimizor(layers, config.optimizer, config.epochs, config.learning_rate, x_train, y_train, x_val, y_val, config.batch_size)

      output_test, accuracy_test, loss_test = predict(x_test.T, y_test, layers)


    print("Test accuracy: ", accuracy_test)
    print("Test loss: ", loss_test)

  return output_test
  """


# In[21]:


get_ipython().system('pip install wandb')


# In[22]:


import wandb


# In[23]:


wandb.login(key='ed57c3903aa24b40dc30a68b77aad62d1489535b')


# In[24]:


sweep_config = {
    'method': 'random',
    'name' : 'sweep cross entropy',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5,10]
        },
        'optimizer': {
        'values': ['adam', 'sgd','nesterov','mgd']
        },
         'h_layers':{
            'values':[2,3,4]
        },
        'activation': {
            'values': ['sigmoid','relu']
        },
        'learning_rate': {
        # a flat distribution between 0 and 0.1
        #'distribution': 'uniform',
        #'min': 0,
        #'max': 0.1
        'values': [0.0001,0.001]
      },
        'neurons': {
            'values': [32,64,128]
        },

        'loss': {
            'values': ['cross_entropy']
        },
        'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms
        #'distribution': 'q_log_uniform_values',
        #'q': 8,
        #'min': 32,
        #'max': 256,
        'values': [16,32,64]
      }

    }
}



# In[25]:


def train():
    config_defaults = {
        'learning_rate': 0.0001,
        'epochs': 10,
        'optimizer': 'adam',
        'loss': 'cross_entropy',
        'init_param': 'Xavier',
        'h_layers' : 3,
        'batch_size':32,
        'neurons' : 128,
        'activation': 'relu'
    }

    #wandb.init(config=config_defaults)
    run =  wandb.init(config=config_defaults)
    run.name=f"{wandb.config.optimizer}_{wandb.config.epochs}_{wandb.config.batch_size}_{wandb.config.h_layers}"
    config = wandb.config
    epochs = config.epochs
    neurons = config.neurons
    learning_rate = config.learning_rate
    neurons = config.neurons
    h_layers = config.h_layers
    activation = config.activation
    batch_size = config.batch_size
    optimizer = config.optimizer
    #output_test = model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train_norm, y_train, x_val, y_val)
    output_test = model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val)




# In[ ]:


import warnings
sweep_id = wandb.sweep(sweep=sweep_config, project='DL_01)')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
wandb.agent(sweep_id, function=train, count=100)


# In[ ]:


"""wandb.agent(sweep_id, function=model_train,count=10) # calls main function for count number of times.
wandb.finish()"""


# In[ ]:


"""activation = 'relu'
batch_size = 64
epochs = 10
h_layers = 4
learning_rate = 0.0001
neurons = 128
optimizer = 'adam'

output_test = model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val)
#wandb.agent(sweep_id, function=model_train,count=10) # calls main function for count number of times.
"""
wandb.finish()


# In[ ]:


"""sweep_id = wandb.sweep(sweep_config, project="Assignment-1_50", entity="swe-rana")
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
wandb.agent(sweep_id, train, count=100)"""


# In[ ]:




