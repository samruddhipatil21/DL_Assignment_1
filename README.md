# DL_Assignment_1

### General info
* It aims at implementing Feedforward neural network from scratch.
* Implementation is done using classes.

#### forward_propagation:
* It takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.
#### back_propagation:
It performs backpropagation and supports following optimisation functions:
* sdg
* momentum
* nesterov
* rmsprop
* adam
#### predict:
* It enables us to predict the labels of the data values on the basis of the trained model. 
#### others functions
* Consists of gradients of activation functions and evaluation metrics

#### command  -

python train.py --wandb_entity myname --wandb_project myprojectname
Arguments to be supported
Name	Default Value	Description
-wp, --wandb_project	myprojectname	Project name used to track experiments in Weights & Biases dashboard
-we, --wandb_entity	myname	Wandb Entity used to track experiments in the Weights & Biases dashboard.
-d, --dataset	fashion_mnist	choices: ["mnist", "fashion_mnist"]
-e, --epochs	1	Number of epochs to train neural network.
-b, --batch_size	4	Batch size used to train neural network.
-l, --loss	cross_entropy	choices: ["mean_squared_error", "cross_entropy"]
-o, --optimizer	sgd	choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
-lr, --learning_rate	0.1	Learning rate used to optimize model parameters
-m, --momentum	0.5	Momentum used by momentum and nag optimizers.
-beta, --beta	0.5	Beta used by rmsprop optimizer
-beta1, --beta1	0.5	Beta1 used by adam and nadam optimizers.
-beta2, --beta2	0.5	Beta2 used by adam and nadam optimizers.
-eps, --epsilon	0.000001	Epsilon used by optimizers.
-w_d, --weight_decay	.0	Weight decay used by optimizers.
-w_i, --weight_init	random	choices: ["random", "Xavier"]
-nhl, --num_layers	1	Number of hidden layers used in feedforward neural network.
-sz, --hidden_size	4	Number of hidden neurons in a feedforward layer.
-a, --activation	sigmoid	choices: ["identity", "sigmoid", "tanh", "ReLU"]
