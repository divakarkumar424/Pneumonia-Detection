import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('pneumonia_2nd_2.h5')

# Define labels
labels = ['Normal', 'Pneumonia']

def predict(image):
    # Resize image and add color channel dimension
    img_array = np.array(image.resize((64, 64)))
    img_array = np.expand_dims(img_array, axis=-1)  # Add color channel dimension
    img_array = np.repeat(img_array, 3, axis=-1)  # Repeat grayscale channel to 3 for RGB
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    prediction = model.predict(img_array)
    return prediction

def main():
    st.title('Pneumonia Detection App🫁')
    st.write('This app uses a deep learning model to detect pneumonia in chest X-ray images.')

    file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = predict(image)
        
        # Check the shape of the prediction array
        if prediction.shape[1] == 1:
            pneumonia_score = prediction[0][0]  # Access the first (and only) element
        else:
            pneumonia_score = prediction[0][1]  # Access the second element
        
        if pneumonia_score > 0.5:
            st.write('Prediction: Pneumonia')
            st.write(f'Confidence: {pneumonia_score}')
        else:
            st.write('Prediction: Normal')
            st.write(f'Confidence: {1 - pneumonia_score}')

if __name__ == '__main__':
    main()
