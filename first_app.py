#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# file management imports
import os  ### only for count of images from dir, can be removed later

# image processing imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import keras
from keras.models import load_model
from tensorflow.keras.models import Sequential
from keras.utils.np_utils import to_categorical 


#for displaying images when predicting class
from PIL import Image 
import cv2
#for rounding up fitting model for steps_per_epoch
import math
#for predicting
# from classify import predict

model = tf.keras.models.load_model('models/MAIN-STARS_model_50_epochs_with_rescaler_zoom-0.2_rotation-45_skew-0.02_flip-hori_verti-02.h5')



#function to predict image
# def predict(uploaded_file): 
#     im = Image.open(uploaded_file)
#     im = tf.keras.preprocessing.image.img_to_array(im)
# #     im = tf.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
#     im = tf.reshape((im,64,64,3))
# #     im = np.expand_dims(im, axis = 0)
#     im = tf.keras.applications.inception_v3.preprocess_input(im)
#     yhat = model.predict(im)
#     label = decode_predictions(yhat)
#     # return highest probability 
#     label = label[0][0]
#     return label 

def testing_image(uploaded_file):
    # loading testing image with the target size for the image
        test_image = Image.open(uploaded_file)
        test_image = test_image.resize((64, 64))
        # converts image into an array
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        # expands array (from converted image) with a new dimension(@ first position) for calculated category
        test_image = np.expand_dims(test_image, axis = 0)
        # making prediction based on test_image and labeling it results
        result = model.predict(x = test_image)
        # printing predictions
        st.write(result)

#         #display image smaller (for checking manually)
#         display_image(image_directory)

        # computing category weither a shape is a rectangle, square or triangle

        if result[0][0] == 1:
            prediction = 'Circle'
            st.write(f'Predicted shape is {prediction}')
        elif result[0][1] == 1:
            prediction = 'Square'
            st.write(f'Predicted shape is {prediction}')
        elif result[0][3] == 1:
            prediction = 'Triangle'
            st.write(f'Predicted shape is {prediction}')
        elif result[0][2] == 1:
            prediction = 'Star'
            st.write(f'Predicted shape is {prediction}')


# In[ ]:

st.title('Shapes - Image Classification w/ CNN')
st.text('Using Streamlit v0.53.0')
activities = ['EDA','Model','About']
choice = st.sidebar.selectbox('Select Activities', activities)
if choice == 'EDA':
    st.subheader('Exploratory Data Analysis')
    st.text('All of the images in my dataset is in .png format.')
    st.text('The first dataset had images with dimensions of 100x100 pixel and the images \n are labeled in 3 folders.')
    st.text('The second dataset had images with dimensions of 28x28 pixel and the images are \n labeled in 3 folders.')
    st.text('Lastly I manually gathered the remaining images for stars. I also had to crop out \n the outlines images of stars')

    st.text('After splitting the dataset(4,438 total images) into Train/Test(90/10 split) of each shape')

    
    #Circle info
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Circle dataset.')
    img = Image.open('collages/EDA_circles.jpg')
    st.image(img, caption='Total images of circles: 1107',width=300)
    st.text('# of images in circle training set: 996')
    st.text('# of images in circle test set: 111')
    #Square info
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Squares dataset.')
    img = Image.open('collages/EDA_squares.jpg')
    st.image(img, caption='Total images of squares: 1115',width=300)
    st.text('# of images in square training set: 1004') 
    st.text('# of images in square test set: 111')
    #Triangle info
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Triangles dataset.')
    img = Image.open('collages/EDA_triangles.jpg')
    st.image(img, caption='Total images of triangles: 1110',width=300)
    st.text('# of images in triangle training set: 999')
    st.text('# of images in triangle test set: 111')
    #Stars info
    st.text('\n')
    st.text('\n')
    st.text('\n')    
    st.text('Sample preview of Stars dataset.')
    img = Image.open('collages/EDA_stars.jpg')
    st.image(img, caption='Total images of stars: 1110',width=300)
    st.text('# of images in stars training set: 1002')
    st.text('# of images in stars test set: 112')
    
elif choice == 'Model':
    st.subheader('Convolutional Neural Network')
    
    uploaded_file = st.file_uploader('Choose an shape image to classify...', type= ['png','jpg'])
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
#         image = image.reshape((64, 64, 3))
        
        st.image(image, caption='Uploaded Image.',width=200, use_column_width=False)
        st.write("")
        st.write("Classifying...")
        testing_image(uploaded_file)
#         label = predict(uploaded_file)
#         st.write('%s (%.2f%%)' % (label[0],label[1], label[2]*100))
#         st.write(prediction)
    
elif choice == 'About':
    st.subheader('About')
    st.text('# of images in circle training set: 996')
    st.text('# of images in circle test set: 111')
    st.text('# of images in square training set: 1004') 
    st.text('# of images in square test set: 111')
    st.text('# of images in triangle training set: 999')
    st.text('# of images in triangle test set: 111')
    st.text('# of images in stars training set: 1002')
    st.text('# of images in stars test set: 112')








# """
# ## My first app
# Here's the total number of images per folder:
# """

# the_image = st.image_input(
