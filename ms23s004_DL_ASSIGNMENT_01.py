#!/usr/bin/env python
# coding: utf-8

# # New Section

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import wandb

# In[2]:


#cross-entropy
def cross_entropy_loss(y, y_hat, i):
  """Returns:
    float: Cross-entropy loss value.
    """
  return -np.log(y_hat[y[i]][0])


def squared_error(y, y_hat, i):
  """
    Compute the squared error loss between the true labels y and the predicted values y_hat.

    Args:
    y (np.array): True labels.
    y_hat (np.array): Predicted values.
    i (int): Index.

    Returns:
    float: Squared error loss value.
    """
  l=len(y)
  return np.mean(np.square(y - y_hat))
  #return np.square(y[i]- y_hat[i])/l

"""def squared_error(y_true, y_pred) :
    loss=0
    l=len(y_true)
    for i in range (l):
     loss+= np.square(y_true[i]-y_pred[i])/l
    return loss
 def cross_entropy(out):
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


# In[3]:


class Layer:

    activation_functions = {
        'tanh': (np.tanh, lambda x: 1 - np.square(np.tanh(x))),
        'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), lambda x: (1 - 1 / (1 + np.exp(-x))) * (1 / (1 + np.exp(-x)))),
        'relu': (lambda x: np.maximum(0, x), lambda x: np.where(x <= 0, 0, 1)),
        'softmax': (lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0, keepdims=True), None)
    }

    def __init__(self, inputs, neurons, activation):
        np.random.seed(33)
        std_dev = np.sqrt(2 / float(inputs + neurons))
        self.W = np.random.normal(0, std_dev, size=(neurons, inputs))
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activation_functions.get(activation)
        self.dW = 0
        self.db = 0


# In[4]:


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
    num_layers = len(layers)

    layers[0].a = np.dot(layers[0].W, h)
    layers[0].h = layers[0].act(layers[0].a)

    for j in range(1, num_layers - 1):
        layers[j].a = np.dot(layers[j].W, layers[j - 1].h)
        layers[j].h = layers[j].act(layers[j].a)

    layers[num_layers - 1].a = np.dot(layers[num_layers - 1].W, layers[num_layers - 2].h)
    layers[num_layers - 1].h = layers[num_layers - 1].act(layers[num_layers - 1].a)

    return layers[num_layers - 1].h


# In[5]:


def backward_propagation(l, y_hat, layers, inp):
    # one-hot vector
    e_l = np.zeros((y_hat.shape[0], 1))
    e_l[l] = 1  # Set the target class index to 1 in the one-hot vector

    # Compute the gradient w.r.t. activation of the last layer (a_L)
    layers[-1].da = -(e_l - y_hat)

    # Backpropagate the gradients through the layers
    for j in range(len(layers) - 1, 0, -1):
        # Update gradients for weights and biases in the current layer
        layers[j].dW += np.dot(layers[j].da, layers[j - 1].h.T)
        layers[j].db += layers[j].da

        # Compute the gradient w.r.t. the hidden layer output (h_j)
        layers[j - 1].dh = np.dot(layers[j].W.T, layers[j].da)
        # Compute the gradient w.r.t. the activation of the hidden layer (a_j)
        layers[j - 1].da = np.multiply(layers[j - 1].dh, layers[j - 1].d_act(layers[j - 1].a))

    # Update gradients for the input layer weights and biases
    layers[0].dW += np.dot(layers[0].da, inp.T)
    layers[0].db += layers[0].da

    return layers  # Return the modified layers list after backward propagation




# In[6]:


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

            # Calculate cost for plotting graph
            cost += squared_error(y_train, h, i)

            # Backpropagation
            backward_propagation(y_train[i], h, layers, inp)

            # Stochastic gradient descent
            if (i + 1) % batch_size == 0:
                for layer in layers:
                    layer.W = layer.W - learning_rate * layer.dW / batch_size
                    layer.b = layer.b - learning_rate * layer.db / batch_size
                    layer.dW = 0
                    layer.db = 0

        costs.append(cost / m)

        # Predict on validation data
        prediction = forward_propagation(x_val.T, layers)

        val_loss = 0
        for i in range(len(y_val)):
            val_loss += squared_error(y_val, prediction[:, i].reshape(10, 1), i)

        val_loss = val_loss / len(y_val)
        prediction = prediction.argmax(axis=0)
        val_accuracy = np.sum(prediction == y_val) / y_val.shape[0]

        # wandb logs
        wandb.log({"epoch": epoch, "loss": costs[len(costs) - 1], "val_accuracy": val_accuracy, "val_loss": val_loss})

        print(f"-----------------epoch {epoch}-----------------")
        print(f"Cost: {cost / m}")
        print(f"Validation accuracy: {val_accuracy}")
        print(f"Validation loss: {val_loss}")


    return costs, layers


# In[7]:


def mgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    gamma = 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):
        # Initialize update parameters for each layer
        for layer in layers:
            layer.update_W = 0
            layer.update_b = 0

        cost = 0

        for i in range(m):
            inp = x_train[i].reshape(784, 1)

            # Feedforward
            h = inp
            h = forward_propagation(h, layers)

            # Calculate cost
            cost += cross_entropy_loss(y_train, h, i)

            # Backpropagation
            backward_propagation(y_train[i], h, layers, inp)

            # Momentum gradient descent
            if (i+1) % batch_size == 0:
                for layer in layers:
                    # Update weights and biases with momentum
                    layer.update_W = gamma * layer.update_W + learning_rate * layer.dW / batch_size
                    layer.update_b = gamma * layer.update_b + learning_rate * layer.dW / batch_size

                    layer.W -= layer.update_W
                    layer.b -= layer.update_b

                    # Reset gradients and updates
                    layer.dW = 0
                    layer.db = 0
                    layer.update_W = 0
                    layer.update_b = 0

        costs.append(cost / m)

        # Predict on validation data
        prediction = forward_propagation(x_val.T, layers)

        val_loss = 0
        for i in range(len(y_val)):
            #val_loss += squared_error(y_val, prediction[:, i].reshape(10, 1), i)
            val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10, 1), i)

        val_loss /= len(y_val)
        prediction = prediction.argmax(axis=0)
        val_accuracy = np.sum(prediction == y_val) / y_val.shape[0]

        # Log metrics
        wandb.log({"epoch": epoch, "loss": costs[-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

        print(f"-----------------epoch {epoch}-----------------")
        print("Cost: ", cost / m)
        print("Validation accuracy: ", val_accuracy)
        print("Validation loss: ", val_loss)

    return costs, layers


# In[8]:


def nesterov(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    gamma = 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):
        # Initialize update parameters for each layer
        for layer in layers:
            layer.update_W = 0
            layer.update_b = 0

        cost = 0

        for i in range(m):
            inp = x_train[i].reshape(784, 1)

            # Feedforward
            h = inp
            h = forward_propagation(h, layers)

            # Calculate cost
            cost += cross_entropy_loss(y_train, h, i)

            # Calculate W lookaheads
            if (i+1) % batch_size == 0:
                for layer in layers:
                    layer.W_lookahead = layer.W - gamma * layer.update_W
                    layer.b_lookahead = layer.b - gamma * layer.update_b

            # Backpropagation
            backward_propagation(y_train[i], h, layers, inp)

            # Nesterov gradient descent
            if (i+1) % batch_size == 0:
                for layer in layers:
                    # Update weights and biases using Nesterov momentum
                    layer.update_W = gamma * layer.update_W + learning_rate * layer.dW / batch_size
                    layer.update_b = gamma * layer.update_b + learning_rate * layer.dW / batch_size

                    layer.W = layer.W_lookahead - layer.update_W
                    layer.b = layer.b_lookahead - layer.update_b

                    # Reset gradients and updates
                    layer.dW = 0
                    layer.db = 0
                    layer.update_W = 0
                    layer.update_b = 0

        costs.append(cost / m)

        # Predict on validation data
        prediction = forward_propagation(x_val.T, layers)

        val_loss = 0
        for i in range(len(y_val)):
            #val_loss += squared_error(y_val, prediction[:, i].reshape(10, 1), i)
             val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10, 1), i)


        val_loss /= len(y_val)
        prediction = prediction.argmax(axis=0)
        val_accuracy = np.sum(prediction == y_val) / y_val.shape[0]

        # Log metrics
        wandb.log({"epoch": epoch, "loss": costs[-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

        print(f"-----------------epoch {epoch}-----------------")
        print("Cost: ", cost / m)
        print("Validation accuracy: ", val_accuracy)
        print("Validation loss: ", val_loss)

    return costs, layers


# In[9]:


import math

def adam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    epsilon, beta1, beta2 = 1e-8, 0.9, 0.99
    t = 0

    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):
        # Initialize Adam parameters for each layer
        for layer in layers:
            layer.m_W, layer.m_b, layer.v_W, layer.v_b = 0, 0, 0, 0
            layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0

        cost = 0

        for i in range(m):
            inp = x_train[i].reshape(784, 1)

            # Feedforward
            h = inp
            h = forward_propagation(h, layers)

            # Calculate cost
            cost += cross_entropy_loss(y_train, h, i)

            # Backpropagation
            backward_propagation(y_train[i], h, layers, inp)

            # Adam gradient descent
            if (i+1) % batch_size == 0:
                t += 1

                for layer in layers:
                    # Update momentum
                    layer.m_W = beta1 * layer.m_W + (1 - beta1) * layer.dW / batch_size
                    layer.m_b = beta1 * layer.m_b + (1 - beta1) * layer.db / batch_size

                    # Update velocity
                    layer.v_W = beta2 * layer.v_W + (1 - beta2) * ((layer.dW / batch_size) ** 2)
                    layer.v_b = beta2 * layer.v_b + (1 - beta2) * ((layer.db / batch_size) ** 2)

                    # Bias correction
                    layer.m_W_hat = layer.m_W / (1 - math.pow(beta1, t))
                    layer.m_b_hat = layer.m_b / (1 - math.pow(beta1, t))
                    layer.v_W_hat = layer.v_W / (1 - math.pow(beta2, t))
                    layer.v_b_hat = layer.v_b / (1 - math.pow(beta2, t))

                    # Update weights and biases
                    layer.W = layer.W - (learning_rate / np.sqrt(layer.v_W_hat + epsilon)) * layer.m_W_hat
                    layer.b = layer.b - (learning_rate / np.sqrt(layer.v_b_hat + epsilon)) * layer.m_b_hat

                    # Reset gradients and updates
                    layer.dW = 0
                    layer.db = 0
                    layer.m_W, layer.m_b, layer.v_W, layer.v_b = 0, 0, 0, 0
                    layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0

        costs.append(cost / m)

        # Predict on validation data
        prediction = forward_propagation(x_val.T, layers)

        val_loss = 0
        for i in range(len(y_val)):
            val_loss += squared_error(y_val, prediction[:, i].reshape(10, 1), i)

        val_loss /= len(y_val)
        prediction = prediction.argmax(axis=0)
        val_accuracy = np.sum(prediction == y_val) / y_val.shape[0]

        # Log metrics
        wandb.log({"epoch": epoch, "loss": costs[-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

        print(f"-----------------epoch {epoch}-----------------")
        print("Cost: ", cost / m)
        print("Validation accuracy: ", val_accuracy)
        print("Validation loss: ", val_loss)

    return costs, layers


# In[10]:


def rmsprop(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    epsilon, beta, decay = 1e-8, 0.9, 0.9  # Hyperparameters
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):
        for layer in layers:
            layer.r_W, layer.r_b = 0, 0

        cost = 0

        for i in range(m):
            inp = x_train[i].reshape(784, 1)

            # Feedforward
            h = inp
            h = forward_propagation(h, layers)

            # Calculate cost
            cost += cross_entropy_loss(y_train, h, i)

            # Backpropagation
            backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

            # RMSprop gradient descent
            if (i+1) % batch_size == 0:
                for layer in layers:
                    layer.r_W = beta * layer.r_W + (1 - beta) * np.square(layer.dW / batch_size)
                    layer.r_b = beta * layer.r_b + (1 - beta) * np.square(layer.db / batch_size)

                    layer.W = layer.W - (learning_rate / np.sqrt(layer.r_W + epsilon)) * (layer.dW / batch_size)
                    layer.b = layer.b - (learning_rate / np.sqrt(layer.r_b + epsilon)) * (layer.db / batch_size)

                    layer.dW = 0
                    layer.db = 0

        costs.append(cost / m)

        # Predict on validation data
        prediction = forward_propagation(x_val.T, layers)

        val_loss = 0
        for i in range(len(y_val)):
            val_loss += squared_error(y_val, prediction[:, i].reshape(10, 1), i)

        val_loss /= len(y_val)
        prediction = prediction.argmax(axis=0)
        val_accuracy = np.sum(prediction == y_val) / y_val.shape[0]

        # Log metrics
        wandb.log({"epoch": epoch, "loss": costs[-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

        print(f"-----------------epoch {epoch}-----------------")
        print("Cost: ", cost / m)
        print("Validation accuracy: ", val_accuracy)
        print("Validation loss: ", val_loss)

    return costs, layers


# In[11]:


def OPTIMIZER(layers, optimizer, epochs, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    if optimizer == "sgd":
        return sgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
    elif optimizer == "mgd":
        return mgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
    elif optimizer == "nesterov":
        return nesterov(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
    elif optimizer == "adam":
        return adam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
    elif optimizer == "rmsprop":
        return adam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
    else:
        print("No optimization algorithm named " + optimizer + " found")
        return "Error", "Error"


# In[12]:


def predict(input, y, layers):
    prediction = forward_propagation(input, layers)

    loss = 0
    for i in range(len(y)):
        #loss += squared_error(y[i], prediction[:, i].reshape(10, 1), i)
        loss += cross_entropy_loss(y, prediction[:, i].reshape(10, 1), i)

    prediction = prediction.argmax(axis=0)
    accuracy = np.sum(prediction == y) / y.shape[0]

    return prediction, accuracy, loss / len(y)


# In[13]:

"""
from keras.datasets import fashion_mnist

(x_train_org, y_train_org), (x_test_org, y_test_org) = fashion_mnist.load_data()
"""

# In[14]:

"""
print("x_train shape: ", x_train_org.shape)
print("y_train shape: ", y_train_org.shape)
"""

# In[15]:

"""
x_train_temp = x_train_org.reshape(x_train_org.shape[0], -1)
y_train_temp = y_train_org
x_test = x_test_org.reshape(x_test_org.shape[0], -1)
y_test = y_test_org
"""

# In[16]:

"""
x_train_norm= x_train_org/255.0
x_test_norm= x_test_org/255.0
"""

# In[16]:





# In[17]:

"""
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.1, random_state=33)
"""

# In[18]:

"""
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_val shape: ", x_val.shape)
print("y_val shape: ", y_val.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)

"""
# In[19]:


def model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val):
    #with wandb.init(config=config):
    #config = wandb.config
    #epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val
    layers= [Layer(x_train.shape[1], neurons, activation)]
    for _ in range(0, h_layers-1):
      layers.append(Layer(neurons, neurons, activation))
      layers.append(Layer(neurons, 10, 'softmax'))

      costs, layers = OPTIMIZER(layers, optimizer, epochs, learning_rate, x_train, y_train, x_val, y_val, batch_size)

      output_test, accuracy_test, loss_test = predict(x_test.T, y_test, layers)


    print("Test accuracy: ", accuracy_test)
    print("Test loss: ", loss_test)

    return output_test


# In[20]:


#get_ipython().system('pip install wandb')


# In[21]:


#import wandb


# In[22]:


#wandb.login(key='ed57c3903aa24b40dc30a68b77aad62d1489535b')


# In[23]:

""""
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
        'values': ['adam','sgd','nesterov','mgd','rmsprop']
        },
         'h_layers':{
            'values':[2,3,4]
        },
        'activation': {
            'values': ['sigmoid','relu','tanh']
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
            'values': ['cross_entropy','squared_error']#,mse]
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
"""


# In[24]:

"""
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

"""


# In[25]:

"""
import warnings
sweep_id = wandb.sweep(sweep=sweep_config, project='DL_01)')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
wandb.agent(sweep_id, function=train, count=1)
"""

# In[26]:


#wandb.finish()


# In[26]:




