import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

train_dir = './PetImages_small/train'
val_dir = './PetImages_small/validation'
test_dir = './PetImages_small/test'
batch_size = 20

# create image data generator
datagen = ImageDataGenerator(rescale=1./255)

# instantiate a Xception pretrained model
conv_base = keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150,150,3)
        )

conv_base.summary()

# extract features from images with pretrained conv base
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count,5,5,2048))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
            directory,
            target_size=(150,150),
            batch_size=batch_size,
            class_mode='binary'
            )
    i = 0
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
val_features, val_labels = extract_features(val_dir,1000)
test_features, test_labels = extract_features(test_dir, 1000)

# reshape features to fit our dense classifier
train_features = np.reshape(train_features, (2000, 5*5*2048))
val_features = np.reshape(val_features, (1000, 5*5*2048))
test_features = np.reshape(test_features, (1000, 5*5*2048))


# build a dense classifier model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu',input_dim=(5*5*2048)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# compile model
model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['acc'])

# train model on training dataset
history = model.fit(
        train_features, train_labels,
        epochs=30,
        batch_size = batch_size,
        validation_data=(val_features, val_labels)
        )

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

