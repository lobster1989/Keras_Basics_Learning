import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
import numpy as np

batch_size = 64

# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# data preprocessing
x_train = x_train.reshape((60000,28,28,1))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000,28,28,1))
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Visualize some pictures
print("Some pictures from dataset: ")
plt.figure(figsize = (6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])
    plt.title('label:' + str(y_train[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()

# build a convolutional model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer="rmsprop", 
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )


# Saperate datasets for train,validate,test
train_dataset = tf.data.Dataset.from_tensor_slices((x_train[:50000], y_train[:50000])).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_train[50000:], y_train[50000:])).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Train the model for 10 epochs using a dataset
print("Fit on Dataset")
history = model.fit(train_dataset, epochs=10,validation_data=val_dataset)

# plot learning curve
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
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

# Evaluate model with test data
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
print("Evaluate model on test Dataset")
loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

predictions = model.predict(val_dataset)
print("Visualize 16 predictions:")
plt.figure(figsize = (8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(x_test[i])
    plt.title('predict:' + str(np.argmax(predictions[i])) + '/label:' + str(y_test[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()







