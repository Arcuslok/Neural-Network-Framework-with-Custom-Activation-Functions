# **Neural Network Framework with Custom Activation Functions**  
*A modular Python implementation of a neural network with support for multiple activation functions (ReLU, Tanh, SoftPlus, Mish, etc.) and backpropagation training.*  

---

## **📌 Features**  
- **Layer-based architecture** with customizable weights, biases, and activation functions.  
- **7+ activation functions** (ReLU, Tanh, Sigmoid, SoftPlus, Mish, Softsign, Sech) with derivatives.  
- **Full training pipeline** (forward/backward propagation, gradient descent).  
- **Clean OOP design** for easy extension.  
- **Numerically stable** implementations with `float32` precision.  

---

## **🚀 Quick Start**  

### **1. Installation**  
```bash
pip install numpy
```

### **2. File Structure**  
```
├── Activation_Functions.py  # All activation functions + derivatives  
├── Neural_Network.py        # Core network implementation (Layer, training, etc.)  
└── Example_Usage.py         # (Optional) Demo script showing how to train a network  
```

### **3. Basic Usage**  
```python
from Neural_Network import Neural_Network, Train, Evaluate
from Activation_Functions import ReLU, Sigmoid

# Create a network: 2 input neurons → 4 hidden neurons → 1 output neuron
network = Neural_Network([2, 4, 1], [ReLU, Sigmoid])

# Train (X = input data, Y = labels)
errors = Train(1000, network, X, Y, learning_rate=0.01)

# Predict
predictions = Evaluate(network, new_data)
```

---

## **📊 Key Components**  

### **1. `Activation_Functions.py`**  
Each activation function is a tuple with:  
- **Name** (e.g., `"ReLU"`)  
- **Function** (e.g., `lambda x: x * (x > 0)`)  
- **Derivative** (e.g., `derivative_relu(x)`)  

| Function  | Range      | Best For                  |  
|-----------|------------|---------------------------|  
| ReLU      | [0, ∞)     | Hidden layers             |  
| Tanh      | (-1, 1)    | Bounded outputs           |  
| Sigmoid   | (0, 1)     | Binary classification     |  
| Mish      | (~-0.3, ∞) | Self-regularized networks |  

### **2. `Neural_Network.py`**  
- **`Layer` Class**: Stores weights, biases, and activation for a layer.  
- **`Training`**: Implements backpropagation with gradient descent.  
- **`Evaluate`**: Forward pass for predictions.  

---

## **📝 Example: XOR Problem**  
```python
# XOR dataset
X = array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = array([[0, 1, 1, 0]]).T

# Train
network = Neural_Network([2, 4, 1], [Tanh, Sigmoid])
Train(5000, network, X, Y, 0.1)

# Test
print(Evaluate(network, X))  # ≈ [0, 1, 1, 0]
```

---

## **⚙️ Customization**  
- **Add new activations**: Define in `Activation_Functions.py` following the same tuple format.  
- **Modify training**: Adjust learning rate or error function in `Neural_Network.py`.  

---

## **📜 License**  
MIT License.  

---

### **🔗 Suggested Repo Names**  
- `Custom-Neural-Network-Framework`  
- `Python-NN-Activation-Functions`  
- `Modular-Neural-Network`  

---

**🎯 Contributions welcome!** Extend with optimizers (Adam, SGD), more layers, or GPU support.  

--- 
