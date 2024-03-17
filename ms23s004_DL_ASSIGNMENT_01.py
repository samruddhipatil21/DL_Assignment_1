
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import wandb
import argparse



"""# **Training And Test Data**"""

#Load the fashion MNIST data


"""# Question 3

"""
def PROPOGATION_FORWARD(self, X_train_batch, weights, biases):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                 : X_train_batch - input 
                 : Weights  - Weights 
                 : biases - Bias  
        """
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
        Y = SOFTMAX(A[str(num_layers - 1)])
        H[str(num_layers - 1)] = Y
        return Y, H, A

(x_train_org, y_train_org), (x_test_org, y_test_org) = fashion_mnist.load_data()

def SIGMOID(x):
     z = np.exp(-x)
     return 1/(1+z)

def exp (y) :
	return (np.e)**y

def d_SIGMOID(x):
  return (1 - SIGMOID(x)) * SIGMOID(x)

def tan_H(x):
  return np.tanh(x)

def d_tan_H(x):
    return 1 - np.square(np.tanh(x))

def RELU(x):
  return np.where(np.asarray(x) > 0, x, 0)

def D_RELU(x):
    return np.where(x <= 0, 0, 1)

def SOFTMAX(x):
    #e_x = np.exp(x)
    #return e_x/e_x.sum()
    # A = np.exp(x)/np.exp(x).sum()
   
    # cache = x
    # return A
    return exp(x) / np.sum(exp(x))


def CROSS_ENTROPY_LOSS(y, y_hat, i):
  return -np.log(y_hat[y[i]][0])


def CROSS_ENTROPY(y, y_hat, i):
        loss = -1 *  np.log(y_hat)
        return loss
        epsilon = 1e-9  # Small constant to avoid log(0)
        m = len(y)
        cost = 0
i=0
for i in range(1):
    """if y_hat[y[i]][0] <= 0:
                 print(f"Error: Invalid probability encountered for index {y[i]}.")
        return None
    cost += -np.log(y_hat[y[i]][0] + epsilon)  # Add epsilon to prevent log(0)"""


class Layer:

    activationFunc = {
        'tanh': (tan_H, d_tan_H),
        'sigmoid': (SIGMOID, d_SIGMOID),
        'relu' : (RELU, D_RELU),
        'softmax' : (SOFTMAX, None)
    }

    def __init__(self, inputs, neurons, activation):

#INITIALIZATION
        np.random.seed(33)
        sd = np.sqrt(2 / float(inputs + neurons))
        self.W = np.random.normal(0, sd, size=(neurons, inputs))
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activationFunc.get(activation)
        self.dW = 0
        self.db = 0


def FF_PROPOGATION(h, LAYERS):
    """
        Returns the neural network LAYERS
        Arguments:
                 : h
                 : LAYERS
        """
    m = len(LAYERS)

    LAYERS[0].a = np.dot(LAYERS[0].W, h)
    LAYERS[0].h = LAYERS[0].act(LAYERS[0].a)

    for j in range(1, m-1):
      LAYERS[j].a = np.dot(LAYERS[j].W, LAYERS[j-1].h)
      LAYERS[j].h = LAYERS[j].act(LAYERS[j].a)

    j+=1
    LAYERS[j].a = np.dot(LAYERS[j].W, LAYERS[j-1].h)
    LAYERS[j].h = SOFTMAX(LAYERS[j].a)


    return LAYERS[m-1].h



def BACK_PROP(l, y_hat, LAYERS, inp):
  """
        Returns the neural network LAYERS
        Arguments:
                 : l
                 : y_hat
                 : LAYERS
                 : inp
        """
  e_l = np.zeros((y_hat.shape[0], 1))
  e_l[l] = 1

  LAYERS[len(LAYERS)-1].da = -(e_l - y_hat)                 
  for j in range(len(LAYERS)-1, 0, -1):

    LAYERS[j].dW += np.dot(LAYERS[j].da, (LAYERS[j-1].h).T)
    LAYERS[j].db += LAYERS[j].da

    LAYERS[j-1].dh = np.dot((LAYERS[j].W).T, LAYERS[j].da)
    LAYERS[j-1].da = np.multiply(LAYERS[j-1].dh, LAYERS[j-1].d_act(LAYERS[j-1].a))

  LAYERS[0].dW += np.dot(LAYERS[0].da, inp.T)
  LAYERS[0].db += LAYERS[0].da

  return LAYERS

def S_G_D(epochs, LAYERS, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    """
        Returns the neural network :costs, LAYERS
        Arguments:
                 epochs, LAYERS, learning_rate, 
                 x_train, y_train, x_val,
                 y_val, batch_size
        """
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      COST_ = 0
      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward NN
        h = inp
        h = FF_PROPOGATION(h, LAYERS)

        # Calulate cost to plot graph
        COST_ += CROSS_ENTROPY_LOSS(y_train, h, i)
        #cost += cross_entropy_loss(y_train, h, i)

        # Backpropagation
        BACK_PROP(y_train[i], h, LAYERS, x_train[i].reshape(784, 1))

        #stocastic gradient decent
        if (i+1) % batch_size == 0:
          for LAYERS in LAYERS:
            LAYERS.W = LAYERS.W - learning_rate * LAYERS.dW/batch_size
            LAYERS.b = LAYERS.b - learning_rate * LAYERS.db/batch_size

            LAYERS.dW = 0
            LAYERS.db = 0

      costs.append(COST_/m)

      prediction = FF_PROPOGATION(x_val.T, LAYERS)

      val_loss = 0
      for i in range(len(y_val)):
       val_loss += CROSS_ENTROPY_LOSS(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Cost: ", COST_/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, LAYERS


def M_G_D(epochs, LAYERS, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    """
        Returns the neural network :costs, LAYERS
        Arguments:
                 epochs, LAYERS, learning_rate, 
                 x_train, y_train, x_val,
                 y_val, batch_size
        """
    GAMMA_VAL = 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for l_number in LAYERS:
        l_number.update_W = 0
        l_number.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)


        h = inp
        h = FF_PROPOGATION(h, LAYERS)

        cost += CROSS_ENTROPY_LOSS(y_train, h, i)

        # BACKPROPOGATION
        BACK_PROP(y_train[i], h, LAYERS, x_train[i].reshape(784, 1))

        #MOMENTUM GD
        if (i+1) % batch_size == 0:
          for l_number in LAYERS:

            l_number.update_W = GAMMA_VAL*l_number.update_W + learning_rate*l_number.dW/batch_size
            l_number.update_b = GAMMA_VAL*l_number.update_b + learning_rate*l_number.dW/batch_size

            l_number.W = l_number.W - l_number.update_W
            l_number.b = l_number.b - l_number.update_b

            l_number.dW = 0
            l_number.db = 0

            l_number.update_W = 0
            l_number.update_b = 0


      costs.append(cost/m)

      PREDICTION_VAL = FF_PROPOGATION(x_val.T, LAYERS)

      val_loss = 0
      for i in range(len(y_val)):
        val_loss += CROSS_ENTROPY_LOSS(y_val, PREDICTION_VAL[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      PREDICTION_VAL = PREDICTION_VAL.argmax(axis=0)
      VAL_ACC =  np.sum(PREDICTION_VAL == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "loss": costs[len(costs)-1], "val_accuracy": VAL_ACC, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Cost: ", cost/m)
      print("Validation accuracy: ", VAL_ACC)
      print("Validation loss: ", val_loss)

    return costs, LAYERS

def NEST_GD(epochs, LAYERS, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    """
        Returns the neural network :costs, LAYERS
        Arguments:
                 epochs, LAYERS, learning_rate, 
                 x_train, y_train, x_val,
                 y_val, batch_size
        """
    GAMMA = 0.9
    m = x_train.shape[0]
    costs = []

    for EPOCHS in range(epochs):

      for l_number in LAYERS:
        l_number.update_W = 0
        l_number.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        h = inp
        h = FF_PROPOGATION(h, LAYERS)

    
        cost += CROSS_ENTROPY_LOSS(y_train, h, i)

       
        if (i+1) % batch_size == 0:
          for l_number in LAYERS:
            l_number.W = l_number.W - GAMMA * l_number.update_W
            l_number.b = l_number.b - GAMMA * l_number.update_b

        # IMPLEMENTATION OF BACKPROPOGATION
        BACK_PROP(y_train[i], h, LAYERS, x_train[i].reshape(784, 1))

        #NGD
        if (i+1) % batch_size == 0:
          for l_number in LAYERS:

            l_number.update_W = GAMMA*l_number.update_W + learning_rate*l_number.dW/batch_size
            l_number.update_b = GAMMA*l_number.update_b + learning_rate*l_number.dW/batch_size

            l_number.W = l_number.W - l_number.update_W
            l_number.b = l_number.b - l_number.update_b

            l_number.dW = 0
            l_number.db = 0

            l_number.update_W = 0
            l_number.update_b = 0

      costs.append(cost/m)

      prediction = FF_PROPOGATION(x_val.T, LAYERS)

      val_loss = 0
      for i in range(len(y_val)):
        val_loss += CROSS_ENTROPY_LOSS(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": EPOCHS, "loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

    
      print("EPOCH "+str(EPOCHS)+"=======")
      print("COST: ", cost/m)
      print("VALIDATION ACCURACY: ", val_accuracy)
      print("VALIDATION LOSS: ", val_loss)

    return costs, LAYERS

def ADAM_(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, BATCH_SIZE):
    """
        Returns the neural network :costs, LAYERS
        Arguments:
                 epochs, LAYERS, learning_rate, 
                 x_train, y_train, x_val,
                 y_val, batch_size
        """
    epsilon, beta1, beta2 = 1e-8, 0.9, 0.99
    t = 0

    m = x_train.shape[0]
    costss = []

    for epoch in range(epochs):

      for LAYERS in layers:
        LAYERS.m_W, LAYERS.m_b, LAYERS.v_W, LAYERS.v_b, LAYERS.m_W_hat, LAYERS.m_b_hat, LAYERS.v_W_hat, LAYERS.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        h = inp
        h = FF_PROPOGATION(h, layers)

        cost += CROSS_ENTROPY_LOSS(y_train, h, i)

        BACK_PROP(y_train[i], h, layers, x_train[i].reshape(784, 1))

    
        if (i+1) % BATCH_SIZE == 0:
          t+=1

          for LAYERS in layers:

            LAYERS.m_W = beta1 * LAYERS.m_W + (1-beta1)*LAYERS.dW/BATCH_SIZE
            LAYERS.m_b = beta1 * LAYERS.m_b + (1-beta1)*LAYERS.db/BATCH_SIZE

            LAYERS.v_W = beta2 * LAYERS.v_W + (1-beta2)*((LAYERS.dW/BATCH_SIZE))**2
            LAYERS.v_b = beta2 * LAYERS.v_b + (1-beta2)*((LAYERS.db/BATCH_SIZE))**2

            LAYERS.m_W_hat = LAYERS.m_W/(1-math.pow(beta1, t))
            LAYERS.m_b_hat = LAYERS.m_b/(1-math.pow(beta1, t))

            LAYERS.v_W_hat = LAYERS.v_W/(1-math.pow(beta2, t))
            LAYERS.v_b_hat = LAYERS.v_b/(1-math.pow(beta2, t))

            LAYERS.W = LAYERS.W - (learning_rate/np.sqrt(LAYERS.v_W_hat + epsilon))*LAYERS.m_W_hat
            LAYERS.b = LAYERS.b - (learning_rate/np.sqrt(LAYERS.v_b_hat + epsilon))*LAYERS.m_b_hat

            LAYERS.dW = 0
            LAYERS.db = 0

            LAYERS.m_W, LAYERS.m_b, LAYERS.v_W, LAYERS.v_b, LAYERS.m_W_hat, LAYERS.m_b_hat, LAYERS.v_W_hat, LAYERS.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0


      costss.append(cost/m)

      #predict on validation data
      PREDiction = FF_PROPOGATION(x_val.T, layers)

      VALIDATION_LOSS = 0
      for i in range(len(y_val)):
      #loss += squared_error(y, prediction[:, i].reshape(10,1), i)
        VALIDATION_LOSS += CROSS_ENTROPY_LOSS(y_val, PREDiction[:, i].reshape(10,1), i)

      VALIDATION_LOSS = VALIDATION_LOSS/len(y_val)
      PREDiction = PREDiction.argmax(axis=0)
      val_accuracy =  np.sum(PREDiction == y_val)/y_val.shape[0]

      print("EPOCH "+str(epoch)+"=======")
      print("COST VALUE: ", cost/m)
      print("VALIDATION ACCURACY: ", val_accuracy)
      print("VALIDATION LOSS: ", VALIDATION_LOSS)

    return costss, layers


def optimizor(LAYERS, OPTIMIZER, epochs, learning_rate, x_train, y_train, x_val, y_val, batch_size):
    
  if OPTIMIZER == "S_G_D":
    return S_G_D(epochs, LAYERS, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif OPTIMIZER == "S_G_D":
    return M_G_D(epochs, LAYERS, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif OPTIMIZER == "NEST_GD":
    return NEST_GD(epochs, LAYERS, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif OPTIMIZER == "ADAM_":
    return ADAM_(epochs, LAYERS, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  else:
    print("INVALID NAME "+OPTIMIZER+" found")
    return "Error", "Error"

def squared_error(y_true, y_pred) :
    """
        Returns Loss
        Arguments:
                : y_true 
                : y_pred
        """
    loss=0
    l=len(y_true)
    for i in range (l):
     loss+= np.square(y_true[i]-y_pred[i])/l
    return loss

def C_ENTROPY(y_true, y_pred):
    loss=0
    l=len(y_true)
    for i in range (l):
     loss+= np.square(y_true[i]-y_pred[i])/l
    return loss


def PREDICTION(input, y, layers):

  prediction = FF_PROPOGATION(input, layers)

  loss = 0
  for i in range(len(y)):
    #loss += squared_error(y, prediction[:, i].reshape(10,1), i)
    loss += CROSS_ENTROPY_LOSS(y, prediction[:, i].reshape(10,1), i)

  prediction = prediction.argmax(axis=0)
  accuracy =  np.sum(prediction == y)/y.shape[0]

  return prediction, accuracy, loss/len(y)

print("x_train shape: ", x_train_org.shape)
print("y_train shape: ", y_train_org.shape)

x_train_temp = x_train_org.reshape(x_train_org.shape[0], -1)
y_train_temp = y_train_org
x_test = x_test_org.reshape(x_test_org.shape[0], -1)
y_test = y_test_org

x_train_norm= x_train_org/255.0
x_test_norm= x_test_org/255.0

x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.1, random_state=33)



def TRAIN_FINAL(EPOCHS, L_RATE, neurons, h_layers, ACTIVATION_Fun, BATCH_SIZE, OPTIMIZER, x_train, y_train, x_val, y_val):
    LAYERS= [Layer(x_train.shape[1], neurons, ACTIVATION_Fun)]
    for _ in range(0, h_layers-1):
      LAYERS.append(Layer(neurons, neurons, ACTIVATION_Fun))
      LAYERS.append(Layer(neurons, 10, 'softmax'))

      costs, LAYERS = optimizor(LAYERS, OPTIMIZER, EPOCHS, L_RATE, x_train, y_train, x_val, y_val, BATCH_SIZE)
    test_output, ACCURACY_TEST, LOSS_TEST = PREDICTION(x_test.T, y_test, LAYERS)


    print("Test_ACCURACY: ", ACCURACY_TEST)
    print("TEST_loss: ", LOSS_TEST)

    return test_output




    