from tensorflow import keras
from tensorflow.keras import layers

#creating a model with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
#units is the number of outputs
#input_shape is the number of inputs

# each layer is a simple computation on the inputs
#with a deepm stack of layers we can solve complex problems

# activation function
# Without activation functions, neural networks can only learn linear relationships. 
# In order to fit curves, we'll need to use activation functions.
# An activation function is simply some function we apply to each of a layer's outputs

# The Rectifier function (also called ReLU) is an activation function defined as
# f(x) = max(0, x)
# Applying a ReLU activation to a linear unit means the output becomes max(0, w * x + b)

#Stacking Dense Layers
# We can create a more complex model by stacking layers.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])