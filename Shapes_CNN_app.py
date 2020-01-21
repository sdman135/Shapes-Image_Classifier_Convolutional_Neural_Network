#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# image processing imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# model imports
import keras
from keras.models import load_model
from tensorflow.keras.models import Sequential
from keras.applications.inception_v3 import preprocess_input

# for displaying images when predicting class
from PIL import Image 




# function to predict image
def predict(uploaded_file): 
    #loading image from the path of image
    im = Image.open(uploaded_file)
    # resizing images to all be the same size    
    im = im.resize((64, 64))    
    # makes sure the image is in RGB (converts all images to have only 3 color channels, png images have 4 color channels)
    im = im.convert(mode='RGB')
    # converts image into an array
    im = tf.keras.preprocessing.image.img_to_array(im)
    # expands array (from converted image) with a new dimension (for calculated category)
    im = np.expand_dims(im, axis = 0)
    im = tf.keras.applications.inception_v3.preprocess_input(im)
    #running image through the model to predict class
    prediction = model.predict(im)
    #showing prediction chart with percentage of each class
    st.write(prediction)
    # computing category weither a shape is a rectangle, square, star or triangle
    if prediction[0][0] > prediction[0][1] + prediction[0][2] + prediction[0][3]:
        st.write(f'Shape is predicted as a Cirle with {"%.1f" % ((prediction[0][0])*100)}% certainty')
    elif prediction[0][1] > prediction[0][0] + prediction[0][2] + prediction[0][3]:
        st.write(f'Shape is predicted as a Square with {"%.1f" % ((prediction[0][1])*100)}% certainty')
    elif prediction[0][2] > prediction[0][0] + prediction[0][1] + prediction[0][3]:
        st.write(f'Shape is predicted as a Star with {"%.1f" % ((prediction[0][2])*100)}% certainty')
    elif prediction[0][3] > prediction[0][0] + prediction[0][1] + prediction[0][2]:
        st.write(f'Shape is predicted as a Triangle with {"%.1f" % ((prediction[0][3])*100)}% certainty')


# In[ ]:

st.title('Shapes - Image Classification w/ CNN')
st.text('Using Streamlit v0.53.0')
# making Sidebar for navigation and options
activities = ['About','EDA','Model']
choice = st.sidebar.selectbox('Select SideBar Activities', activities)
                 
# Sidebar content for About
if choice == 'About':
    st.subheader('Mission Statement')
    st.text('I wanted to make a image classifier using a convolutional neural network model. \nConvolutional neural networks is one of the princable “deep learning” models that \nworks great with images and videos. \n')
    st.text('\n')
    st.text('\n')  
    st.text('I wanted to make a model and have a system where I can feed a image, I will be taken \nlive and, hopefully predict the correct class(Shape). \n')
    img = Image.open('images_for_presentation/shapesorting7square.jpg')
    st.image(img, caption="I'm basically doing a coding version of this kids toy",width=300)
    st.text('') 
    st.text('')

                 
# Sidebar content for EDA
elif choice == 'EDA':
    st.subheader('Exploratory Data Analysis')
    st.text('All of the images in my dataset is in .png format.')
    st.text('The first dataset had images with dimensions of 100x100 pixel and the images \nare labeled in 3 folders.')
    st.text('The second dataset had images with dimensions of 28x28 pixel and the images are \nlabeled in 3 folders.')
    st.text('Lastly I manually gathered the remaining images for stars. I also had to crop out \nthe outlines images of stars.')

    st.text('After splitting the dataset(4,438 total images) into Train/Test(90/10 split) of each shape')

    
    #Circle dataset information:
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Circle dataset.')
    img = Image.open('collages/EDA_circles.jpg')
    st.image(img, caption='Total images of circles: 1107',width=300)
    st.text('# of images in circle training set: 996')
    st.text('# of images in circle test set: 111')
    #Square dataset information:
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Squares dataset.')
    img = Image.open('collages/EDA_squares.jpg')
    st.image(img, caption='Total images of squares: 1115',width=300)
    st.text('# of images in square training set: 1004') 
    st.text('# of images in square test set: 111')
    #Triangle dataset information:
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Triangles dataset.')
    img = Image.open('collages/EDA_triangles.jpg')
    st.image(img, caption='Total images of triangles: 1110',width=300)
    st.text('# of images in triangle training set: 999')
    st.text('# of images in triangle test set: 111')
    #Stars dataset information:
    st.text('\n')
    st.text('\n')
    st.text('\n')    
    st.text('Sample preview of Stars dataset.')
    img = Image.open('collages/EDA_stars.jpg')
    st.image(img, caption='Total images of stars: 1110',width=300)
    st.text('# of images in stars training set: 1002')
    st.text('# of images in stars test set: 112')

# Sidebar content for Model
elif choice == 'Model':
    #loading trained model for predicting
    model = tf.keras.models.load_model('models/MAIN-STARS_model_50_epochs_with_rescaler_zoom-0.2_rotation-45_skew-0.02_flip-hori_verti-02-main.h5')

    st.subheader('Convolutional Neural Network - Realtime Predictor')
    
    uploaded_file = st.file_uploader('Choose an image of a shape to classify...', type= ['png','jpg'])
    if uploaded_file is not None:
        
        im = Image.open(uploaded_file)
#         im = tf.keras.preprocessing.image.reshape((image,64,64,3))
        
        
        st.image(im, caption='Uploaded Image.',width=200, use_column_width=False)
        st.write("")
        st.write("Classifying at")
        predict(uploaded_file)

    









# """
# ## My first app
# Here's the total number of images per folder:
# """

# the_image = st.image_input(
