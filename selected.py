import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import streamlit as st 
import numpy as np

st.title("MNIST Classification")

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


st.write("Training data shape:", X_train.shape)

# Add a selectbox to select the sample image
sample_image_index = st.selectbox("Select a sample image:", range(len(X_test)))

# Display the selected sample image
st.write("Sample Image:")
plt.imshow(X_test[sample_image_index])
st.pyplot(plt)

X_train = X_train / 255
X_test = X_test / 255

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
st.write("Test Accuracy:", accuracy)

st.write("Model Accuracy:")
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Training Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_title('Model Accuracy')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()
st.pyplot(fig)

    # Make a prediction
input_image = X_test[sample_image_index].reshape(1, 28, 28)
prediction = model.predict(input_image)
predicted_class = np.argmax(prediction)
    
    # Display the prediction
st.write("Predicted Class:", predicted_class)

