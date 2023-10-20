import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert A.shape == Z.shape

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert dZ.shape == Z.shape

    return dZ


def initialize_parameters_rand(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = tf.Variable(
            np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01, dtype=tf.float64
        )
        parameters["b" + str(l)] = tf.Variable(
            np.zeros((layer_dims[l], 1)), dtype=tf.float64
        )

        assert parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layer_dims[l], 1)

    return parameters


def initialize_parameters_from_model(layer_dims, file):
    """
    Arguments:
    layer_dims -- dimensions of layers
    file -- name of the file containing model weights

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    model = Sequential()
    NHlayers = len(layer_dims) - 1
    for ilayer in range(NHlayers):
        model.add(Dense(layer_dims[ilayer], activation="relu"))
    model.add(Dense(layer_dims[-1], activation="sigmoid"))
    model.build(input_shape=(1, 1))
    model.load_weights(file)

    parameters = {}
    model_weights = model.get_weights()
    L = len(model_weights) // 2  # number of layers in the network
    for l in range(L):
        parameters["W" + str(l + 1)] = tf.Variable(
            np.transpose(model_weights[2 * l]), dtype=tf.float64
        )
        parameters["b" + str(l + 1)] = tf.Variable(
            model_weights[2 * l + 1].reshape(-1, 1), dtype=tf.float64
        )

    return parameters

def save_model_weights(parameters, model_name, directory_path = None):
	L = len(parameters) // 2  # number of layers in the NN
	
	layer_dims = []
	weights = []
	for l in range(L):
		layer_dims.append(parameters['W' + str(l + 1)].shape[0] )
		weights.append(np.transpose(parameters['W' + str(l + 1)].numpy()))
		weights.append(parameters['b' + str(l + 1)].numpy().reshape(-1, ))
	
	model = Sequential()
	NHlayers = len(layer_dims) - 1
	for ilayer in range(NHlayers):
		model.add(Dense(layer_dims[ilayer], activation="relu"))
	model.add(Dense(layer_dims[-1], activation="sigmoid"))
	model.build(input_shape=(1, 1))
	model.set_weights(weights)
	
	model.save_weights(directory_path + '/' + model_name)
	
	return None
	

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)

    return A, cache


def model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu"
        )
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid"
    )
    caches.append(cache)

    assert AL.shape == (parameters["W" + str(L)].shape[0], X.shape[1])

    return AL, caches


def linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache

    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1).reshape((b.shape[0], 1))
    dA_prev = np.dot(tf.transpose(W), dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def model_backward(AL, dAL, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = linear_activation_backward(dAL, caches[L - 1], activation="sigmoid")
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = current_cache

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = linear_activation_backward(
            grads["dA" + str(l + 1)], caches[l], activation="relu"
        )
        dA_prev_temp, dW_temp, db_temp = current_cache
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


'''    
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = tf.math.subtract(parameters['W' + str(l+1)], learning_rate * grads['dW' + str(l+1)] )
        parameters["b" + str(l+1)] = tf.math.subtract(parameters["b" + str(l+1)], learning_rate * grads['db' + str(l+1)] )
    return parameters
'''


def update_parameters(parameters, grads, optimizer):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward
    optimizer - Tensorflow optimizer

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network
    param_list = []
    grad_list = []

    for l in range(L):
        param_list.append(parameters["W" + str(l + 1)])
        param_list.append(parameters["b" + str(l + 1)])
        grad_list.append(grads["dW" + str(l + 1)])
        grad_list.append(grads["db" + str(l + 1)])

    optimizer.apply_gradients(grads_and_vars=zip(grad_list, param_list))

    # Reassign updated parameters
    for l in range(L):
        parameters["W" + str(l + 1)] = param_list[2 * l]
        parameters["b" + str(l + 1)] = param_list[2 * l + 1]

    return parameters
