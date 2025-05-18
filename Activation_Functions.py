from numpy import exp, log
from warnings import filterwarnings
filterwarnings('ignore')

#--------------- Neural Network Functions --------------

# Lambda expressions are ideally used when we need to do something simple and are more interested in getting the job done quickly rather than formally naming the function. We'll cover them later in the Neural Network.

def derivative_relu(x):
  x[x<=0] = 0.0
  x[x>0] = 1.0
  return x

Relu = (
  # Name of the Activation Function
  lambda: "Relu",
  
  # Activation Function
  lambda x: x * (x > 0),

  # Derivative Activation Function
  lambda x: derivative_relu(x)

)


# Hyperbolic Tangent is generally used in hidden layers. The output of tanh lies between -1 and +1. Therefore, the mean of the hidden layers remains close to 0. This makes hidden layer learning converge quickly.
Tanh = (
  # Name of the Activation Function
  lambda: "Tanh",
  
  # Activation Function
  lambda x: (1.0 - (2.0 / (1.0 + exp(2.0*x)))),

  # Derivative Activation Function
  lambda x: ((4.0 * exp(2.0*x))/(exp(2.0*x) + 1.0)**2.0)
)

# SoftPlus is generally used in the case of binary classification, where the output is either 0 or 1.
SoftPlus = (
  # Name of the Activation Function
  lambda: "SoftPlus",
  
  # Activation Function
  lambda x: (log(1.0 + exp(x))),
  
  # Derivative Activation Function
  lambda x: (exp(x) / (1.0 + exp(x)))
)

# Mish is a self-regularized function that should be used in the hidden layer and is continuously differentiable with infinite order. It has the property of not being bounded either above or below and is more accurate than other activation functions such as Swish and ReLU.
Mish = (
  # Name of the Activation Function
  lambda: "Mish",
  
  # Activation Function
  lambda x: (x * Tanh[1](SoftPlus[1](x))),
  
  # Derivative Activation Function
  lambda x: (Sech[1](SoftPlus[1](x))**2.0 * x * Sigmoid[1](x) + (Mish[1](x) / x))
)

# Sigmoid is generally used in the case of binary classification where the output is either 0 or 1. As the output of sigmoid lies between 0 and 1, it can be easily predicted that the result will be 1 if the value is greater than 0.5 and 0 otherwise.
Sigmoid = (
  # Name of the Activation Function
  lambda: "Sigmoide",
  
  # Activation Function
  lambda x: (1.0 / (1.0 + exp(-x))),
  
  # Derivative Activation Function
  lambda x: (exp(-x)/(1.0 + exp(-x))**2.0)
)

# Softsign is an activation function for neural networks. Although the tanh and softsign functions are closely related, tanh converges exponentially, while softsign converges polynomially.
Softsign = (
  # Name of the Activation Function
  lambda: "Softsign",
  
  # Activation Function
  lambda x: (x / (1.0 + abs(x))),
  
  # Derivative Activation Function
  lambda x: (1.0 / ((1.0 + abs(x))**2.0))
)

# Sech produces outputs scaled to [0,1]. The output decreases and closes to neutral as x approaches infinity. However, it will never produce a 0 output, even for very large inputs, except for ±∞.
Sech = (
  # Name of the Activation Function
  lambda: "Sech",
  
  # Activation Function
  lambda x: (2.0 / (exp(x) + exp(-x))),
  
  # Derivative Activation Function
  lambda x: (-(2.0*(exp(x) - exp(-x))) / (exp(x) + exp(-x))**2.0)
)
