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

#OVERFITTING AND UNDERFITTING
#The information in the training data can be divided into two parts:
#Signal: The part of the information that is actually useful for making predictions.
#Noise: A random (and usually small) error term caused by things like messy data or inconsistent relationships between the features and the target.

#when the training and validation curves are graphed together, there are three common situations:
#The model is too overfit: This is indicated by validation loss that is consistently higher than training loss.
#Overfitting the training set is when the loss is not as low as it could be because the model learned too much noise.
#The model is too underfit: This is indicated by both training and validation loss that is very high.
#Underfitting the training set is when the loss is not as low as it could be because the model hasn't learned enough signal
#The model is neither overfit nor underfit. The sweet spot in the middle is where training and validation loss are both low.

#if the models curve go down together that means the model is learning from the signal data and the loss gets decreaed
#if the models curve go up together that means the model is learning from the noise data and the loss gets increased
#the gap between the curves tells us how much noise the model has learned.
#you cant ideally train a model so there will always be noise with signals but you have to ballence the ratio between them

#Capacity

#A model's capacity refers to the size and complexity of the patterns it is able to learn.
#You can increase the capacity of a network either by making it wider (more units to existing layers) or by making it deeper (adding more layers).

model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

wider = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])

deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

