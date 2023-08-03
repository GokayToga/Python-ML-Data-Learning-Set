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

#The loss function and the optimizer - stochastic gradient descent
#The loss function measures the disparity between the the target's true value and the value the model predicts
#The optimizer is an algorithm that adjusts the weights to minimize the loss.
#stoachastic gradient descent is a common optimizer and the steps are:
#Sample some training data and run it through the network to make predictions.
#Measure the loss between the predictions and the true values.
#Finally, adjust the weights in a direction that makes the loss smaller.

#each sample of training data is called a batch
#a complete round of the training data is called an epoch
# the learning rate determines the size of the step we take in each epoch

#adds an optimizer and loss function for training
model.compile(
    optimizer="adam",
    loss="mae",
)

#wine quality example

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([#creeate a sequential model
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(#compile the model
    optimizer='adam',
    loss='mae',
)
history = model.fit(#train the model
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

#plotting the model loss curve can help us better understand our model
import pandas as pd

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();

#With the learning rate and the batch size, you have some control over:
#How long it takes to train a model
#How noisy the learning curves are
#How small the loss becomes