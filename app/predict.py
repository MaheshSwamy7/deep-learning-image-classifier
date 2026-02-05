import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model/cnn_model.h5')

# Load and preprocess the image
img = image.load_img('test.jpg', target_size=(150, 150))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)

# Print result
if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
