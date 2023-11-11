import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import streamlit as st
from keras.datasets import mnist
from PIL import Image


# Load MNIST handwritten digit data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Convert y_train into one-hot format
temp = []
for i in range(len(y_train)):
    temp.append(to_categorical(y_train[i], num_classes=10))

y_train = np.array(temp)

# Convert y_test into one-hot format
temp = []
for i in range(len(y_test)):
    temp.append(to_categorical(y_test[i], num_classes=10))

y_test = np.array(temp)

# Create simple Neural Network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# Train the Neural Network model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Making predictions using our trained model
X_test1 = X_test[0].reshape(1, 28, 28)

st.header("Digit Identification App", divider='rainbow')
left_column, right_column = st.columns(2)
im = left_column.image(X_test[0], width=200)
if right_column.button("Predict"):
    predictions = model.predict(X_test1)
    predictions = np.argmax(predictions, axis=1)
    right_column.write(f"The digit in the image seems to be {predictions}")
