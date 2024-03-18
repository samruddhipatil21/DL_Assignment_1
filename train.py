from ms23s004_DL_ASSIGNMENT_01 import *
import argparse
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.datasets import fashion_mnist

def argument_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL_01)')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='samruddhipatil2526')
    parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', choices = ["mnist", "fashion_mnist"],type=str, default='fashion_mnist')
    parser.add_argument('-a', '--activation', help='choices: ["sigmoid","relu","tanh"]', choices = ["sigmoid","relu","tanh"],type=str, default="relu")
    parser.add_argument('-o', '--optimizer', help = 'choices: ["adam","sgd","nesterov","mgd","rmsprop"]', choices = ["adam","sgd","nesterov","mgd","rmsprop"],type=str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.001)
    parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
    parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=4)
    parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', type=int, default=128)
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=64)
    args = vars(parser.parse_args())
    
    return args

if __name__=='__main__':
	
	args = argument_parser()
  
   
wandb.login(key="ed57c3903aa24b40dc30a68b77aad62d1489535b")
#wandb.init(project=args['wandb_project'], entity=args['wandb_entity'])
wandb.init(project=args['wandb_project'], entity=args['wandb_entity'])

if args['dataset']=="fashionmnist":
        (x_train_org, y_train_org), (x_test_org, y_test_org) = fashion_mnist.load_data()
    
else:
        (x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
        

x_train_temp = x_train_org.reshape(x_train_org.shape[0], -1)
y_train_temp = y_train_org
x_test = x_test_org.reshape(x_test_org.shape[0], -1)
y_test = y_test_org
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.1, random_state=33)
    

activation = args['activation']
batch_size = args['batch_size']
epochs = args['epochs']
h_layers = args['num_layers']
learning_rate = args['learning_rate']
neurons = args['hidden_size']
optimizer = args['optimizer']
output_test = model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val)

wandb.finish()