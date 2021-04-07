import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np

# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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



# Build a simple model
inputs = keras.Input(shape=(28, 28))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="adam", 
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )

# Train the model for 1 epoch from Numpy data
batch_size = 64
#print("Fit on NumPy data")
#history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

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







