# Practise word-embedding with Keras

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers 
import tensorflow.keras.models as model
import matplotlib.pyplot as plt
import numpy as np


# download dataset from IMDB
(x_train, y_train),(x_test, y_test) = keras.datasets.imdb.load_data(num_words = 10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, 200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, 200)

# create model with word embedding layer
model = keras.models.Sequential()
model.add(layers.Embedding(10000, 8, input_length=200)) # input_dim, output_dim, input_length
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# build model
history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# plot learning curve
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1,len(loss_values) + 1)

plt.figure(figsize = (6,6))
plt.subplot(2,1,1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title("Training and validation loss")
plt.ylabel('Loss')
plt.legend()
plt.subplot(2,1,2)
plt.plot(epochs, acc, 'ro', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title("Training and validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

