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

#Early Stopping

#When the model is too eagerly learning noise, the validation loss may start to increase during training 
#to prevent this we can use early stopping, whenever it seems the validation loss isnt increasing anymore we can stop the training

#So besides preventing overfitting from training too long, early stopping can also prevent underfitting from not training long enough
#we include early stopping in our training through a callback.

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)
#example countinuing

from tensorflow import keras
from tensorflow.keras import layers, callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)

#after creatin early stopping we add it to the fit function

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))


#there is special layers that doesnt contain any neurons themselves but add some functionality to the model
#there is two special layers in hand, the dropout layer and the batch normalization layer

#DROP OUT LAYERS
#the idea behind dropout is to randomly drop some of the neurons in the layer during training
#this prevents the model from relying too much on any one neuron, so it learns a more robust set of features

keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])

#BATCH NORMALIZATION LAYERS
#this helps correct training that is slow or unstable
#A batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, 
#and then also putting the data on a new scale with two trainable rescaling parameters.

keras.Sequential([
    # ...
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    # ...
])

keras.Sequential([
    # ...
    layers.Dense(16),
    layers.BatchNormalization(),
     layers.Activation('relu'),
    # ...
])

# if you add it as the first layer of your network it can act as a kind of adaptive preprocessor, 
# standing in for something like Sci-Kit Learn's StandardScaler.

#when adding dropout, you may need to increase the number of units in your Dense layers.

model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

#training is the same

#Binary Classification
#A binary classifier is simply a model which can decide between two outcomes—usually yes/no, or 1/0.

#Accuracy is the ratio of correct predictions to total predictions: accuracy = number_correct / total
#eventhogh accuracy is a good metric for classification, it can not be used as an actual loss function
#so we have to choose a substitute loss function, the most common one is binary cross-entropy function

#Cross-entropy is a sort of measure for the distance from one probability distribution to another.

#To covert the real-valued outputs produced by a dense layer into probabilities,
#we attach a new kind of activation function, the sigmoid activation.

#everything is the same but In the final layer include a 'sigmoid' activation so that the model will produce class probabilities.

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))