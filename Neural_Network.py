from numpy import random, mean, sqrt, round, float32

# Structure of the Layers of our Neural Network
class Layer():

  # When our structure is called, it will ask for:
  # - The Layer Number
  # - The Number of Neurons in the Previous Layer
  # - The Number of Neurons in the Layer
  # - The Activation Function to be applied to the Layer
  
  def __init__(self, layer_number, n_neurons_previous_layer, n_neurons, activation_function):
    
    # The Layer Number is stored in "Number"
    self.Number = layer_number
    
    # The Activation Function that will be applied to the Layer is stored in "Activation_Function"
    self.Activation_Function = activation_function

    # A bias matrix is ​​created, which is 1 row by the number of neurons in the current layer, initialized to zeros.
    self.Bias  = random.rand(1, n_neurons).astype(float32) #, dtype=float32)#.reshape(1,n_neurons)

    # A weight matrix is ​​created, where the number of neurons in the previous layer is multiplied by the number of neurons in the current layer, initialized to zeros.
    self.Weights = (random.rand(n_neurons_previous_layer, n_neurons)*sqrt(2/n_neurons_previous_layer)).astype(float32) #, dtype=float32).reshape(n_neurons_previous_layer,n_neurons)

  def __repr__(self):
    return "\n Hidden Layer %s --- Activation Function: %s \n\n Bias: \n%s \n\n Weights: \n %s " % (self.Number, self.Activation_Function[0](), self.Bias, self.Weights)


# Function to create the Neural Network
def Neural_Network(neurons, activation_functions):

  neural_network = []

  for step in range(len(neurons)-1):
    if step == len(neurons)-2:
      capa = Layer("Salida", neurons[step], neurons[step+1], activation_functions[step])
      neural_network.append(capa)
      
    else:
      capa = Layer(step+1, neurons[step], neurons[step+1], activation_functions[step])
      neural_network.append(capa)
  
  return neural_network

# Error Function
def Error(predicted_output, activation_result):

  # We calculate the derivative of the function with respect to the output: ∂E/∂a^2 = (y_i - a_i)
  derivative = predicted_output - activation_result

  # We calculate the Error (Mean Square Error) : Σ(y_i - a_i)^2 / 2
  error = mean(derivative ** 2)

  return (error, derivative)

# Function to create the Predict within our Network
def Evaluate(neural_network, input):
  # Output will save the result of each Layer
  # In Layer 1, the result is the Input Value
  Output = [input]

  # Forward propagation
  for num_layer in range(0, len(neural_network)):
    
    # Propagation Function: Σ(Inputs * Weights) + Bias
    sum_weight = Output[-1] @ neural_network[num_layer].Weights + neural_network[num_layer].Bias

    # Activation Function:
    activation = neural_network[num_layer].Activation_Function[1](sum_weight)

    # We include the result of the Layer to Output
    Output.append(activation)

  return Output[-1]

# Function to train the Network
def Training(neural_network, input, expected_output, learning_rate):

  # Output will save the result of each Layer
  # In Layer 1, the result is the Input Value
  Output = [input]

  # Forward propagation
  for num_layer in range(0, len(neural_network)):
    
    # Propagation Function: Σ(Inputs * Weights) + Bias
    sum_weight = Output[-1] @ neural_network[num_layer].Weights + neural_network[num_layer].Bias

    # Activation Function:
    activation = neural_network[num_layer].Activation_Function[1](sum_weight)

    # We include the result of the Layer to Output
    Output.append(activation)
  

  # Backpropagation
  backpropagation = list(range(0, len(Output)-1))
  backpropagation.reverse()
  
  # We will save the Layer Error in delta δ
  delta = []

  # We start from the Output of the Neural Network to the Input of the Neural Network 
  for layer in backpropagation:

    # Layer i Activation Function:
    activation = Output[layer+1]

    # If the Layer is the last in the Neural Network
    if layer == backpropagation[0]:

      # δ_layer = ∂Error/∂Activation * ∂Activation/∂Derivative_Activation
      error_capa = Error(activation, expected_output)[1] * neural_network[layer].Activation_Function[2](activation)
      delta.append(error_capa)
    
    else:
      # δ_layer = δ_previous_layer * Previous_layer_weights * Activation_Derivative(activation)
      error_capa = delta[-1] @ Previous_layer_weights * neural_network[layer].Activation_Function[2](activation)
      delta.append(error_capa)
  
    Previous_layer_weights = neural_network[layer].Weights.transpose()

    # Gradient Descent: Calculates the inverse of the gradient to determine what values ​​the hyperparameters (weights and biases) should take.
    # How far we move downward will depend on another hyperparameter: the learning rate.

    # Layer_bias = Layer_bias - (Previous_Σδ_layer / Previous_δ_layer_number) * learning_rate
    neural_network[layer].Bias = neural_network[layer].Bias - mean(delta[-1], axis = 0, keepdims = True) * learning_rate

    # Layer_weights = Layer_weights - (transposed_layer_activation * δ_previous_layer) * learning_rate
    neural_network[layer].Weights = neural_network[layer].Weights - Output[layer].transpose() @ delta[-1] * learning_rate

  return Output[-1]

# Function to train the network multiple times
def Train(iterations, neural_network, input, expected_output, learning_rate):

  iterations = list(range(0,iterations))
  errors = []
  predictions = []
    
  for iteracion in iterations:
    ronda = Training(neural_network, input, expected_output, learning_rate)
    predictions.append(ronda)
    error_temporal = Error(round(predictions[-1]), expected_output)[0]
    errors.append(error_temporal)
    print("Error: {}".format(errors[-1]), end="\r")

  return errors
