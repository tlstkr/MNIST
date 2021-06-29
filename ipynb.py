#!/usr/bin/env python
# coding: utf-8

# In[67]:


import tensorflow as tf
import numpy as np

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))


model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

print(x_train[0])

model.save('epic_num_reader.model')

new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict([x_test])

print(predictions)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()






