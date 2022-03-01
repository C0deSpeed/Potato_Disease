import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from PIL import Image,ImageOps
import streamlit as st

model = tf.keras.models.load_model('C:\Potato_disease\Models\Version_2')
class_names = ['Early Blight', 'Late Blight', 'Healthy']

def image_processing(img):
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    image = img
    size = (256,256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array
    
    return data

def predict(img):
    prediction = class_names[np.argmax(model.predict(img)[0])]
    return prediction

def main():
    st.title('Potato Disease Prediction')
    html_temp = """
    <h5 style="color:white;text-align:left;">By Rohan Sharma</h5>
    <div style="background-color:darkblue;padding:10px">
    <h2 style="color:white;text-align:center;">Enter Information Below</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        image = image_processing((Image.open(uploaded_file)))
        st.image(Image.open(uploaded_file))
        if st.button('Predict'):
            st.success("Classifying...")
            st.write(predict(image))
        
        

if __name__=='__main__':
    main()