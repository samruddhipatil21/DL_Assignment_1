from ms23s004_DL_ASSIGNMENT_01 import *
import argparse
import wandb

def argument_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL_01)')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='')
    parser.add_argument('-a', '--activation', help='choices: ["SIGMOID","RELU"]', choices = ["SIGMOID","RELU"],type=str, default='relu')
    parser.add_argument('-o', '--optimizer', help = 'choices: ["ADAM_", "S_G_D","NEST_GD","mgd"]', choices = ["ADAM_", "S_G_D","NEST_GD","mgd"],type=str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.001)
    parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
    parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=4)
    parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', type=int, default=128)
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=64)
    args = vars(parser.parse_args())
    return args



if __name__=='__main__':
	
	args = argument_parser()
   
	wandb.login(relogin="",key="ed57c3903aa24b40dc30a68b77aad62d1489535b")
	wandb.init(project=args['wandb_project'], entity=args['wandb_entity'])

	ACTIVATION_Fun = args['activation']
	BATCH_SIZE = args['batch_size']
	EPOCHS = args['epochs']
	h_layers = args['num_layers']
	L_RATE = args['learning_rate']
	neurons = args['hidden_size']
	OPTIMIZER = args['optimizer']
	output_test = TRAIN_FINAL(LAYERS, OPTIMIZER, EPOCHS, L_RATE, x_train, y_train, x_val, y_val, BATCH_SIZE)

	wandb.finish()