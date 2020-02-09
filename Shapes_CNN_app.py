#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#image processing imports
import tensorflow as tf
#for converting image into array
from tensorflow.keras.preprocessing import image

#imports for model
import keras
from keras.models import load_model
from tensorflow.keras.models import Sequential
from keras.applications.inception_v3 import preprocess_input

# for displaying images when predicting class
from PIL import Image 



################
#function to predict image
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
    st.write("\n")
    #showing prediction chart with percentage of each class
    st.write(pd.DataFrame(prediction, columns = ['Circle','Square','Star','Triangle'], index = ['Probability']))
    st.write("\n")
    # computing category weither a shape is a rectangle, square, star or triangle
    if prediction[0][0] > prediction[0][1] + prediction[0][2] + prediction[0][3]:
        st.write(f'Shape is predicted as a Circle with {"%.2f" % ((prediction[0][0])*100)}% certainty')
    elif prediction[0][1] > prediction[0][0] + prediction[0][2] + prediction[0][3]:
        st.write(f'Shape is predicted as a Square with {"%.2f" % ((prediction[0][1])*100)}% certainty')
    elif prediction[0][2] > prediction[0][0] + prediction[0][1] + prediction[0][3]:
        st.write(f'Shape is predicted as a Star with {"%.2f" % ((prediction[0][2])*100)}% certainty')
    elif prediction[0][3] > prediction[0][0] + prediction[0][1] + prediction[0][2]:
        st.write(f'Shape is predicted as a Triangle with {"%.2f" % ((prediction[0][3])*100)}% certainty')

######################
# In[2]:

st.title('Shapes - Image Classification w/ CNN')
st.text('Using Streamlit v0.53.0')
# making Sidebar for navigation and options
activities = ['Mission Statement','EDA','Predictor','About']
choice = st.sidebar.selectbox('SideBar Navigation', activities)
                 
# Sidebar content for About
if choice == 'Mission Statement':
    st.subheader('Mission Statement')
    st.text('I wanted to make a image classifier using a Convolutional Neural Network model. \nConvolutional Neural Networks are one of the princable “deep learning” models that \nworks amazingly with images and videos. \n')
    st.text('\n')
    st.text('\n')  
    st.text('I wanted to make a model and have a system where I can feed an image of a shape, I \nwill be taking live and, hopefully predict the correct Shape (class). \n')
    # simple images to convey 'shape sorting'
    img01 = Image.open('ignore_files/images_for_presentation/shapesorting.jpg')
    img02 = Image.open('ignore_files/images_for_presentation/wooden-baby-shape-puzzle-toy.png')

    st.image([img01,img02],width=300)
    st.text(" I'm basically doing a pythonic coding version of this children's game.") 
    st.text('')

                 
# Sidebar content for EDA
elif choice == 'EDA':
    st.subheader('Exploratory Data Analysis')
    st.text('All of the images in my dataset is in .png format.')
    st.text('The first dataset had images with dimensions of 100x100 pixel and the images \nare labeled in 3 folders.')
    st.text('The second dataset had images with dimensions of 28x28 pixel and the images are \nlabeled in 3 folders.')
    st.text('Lastly I manually gathered the remaining images for stars. I also had to crop out \nthe outlines images of stars.')
    st.text('\n')
    st.text('After splitting the dataset(4,438 total images) into Train/Test(90/10 split) of each shape')

    
    #Circle dataset information:
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Circle dataset.')
    img = Image.open('collages/EDA_circles.png')
    st.image(img, caption='Total images of circles: 1107',width=300)
    st.text('# of images in circle training set: 996')
    st.text('# of images in circle test set: 111')
    #Square dataset information:
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Squares dataset.')
    img = Image.open('collages/EDA_squares.png')
    st.image(img, caption='Total images of squares: 1115',width=300)
    st.text('# of images in square training set: 1004') 
    st.text('# of images in square test set: 111')
    #Triangle dataset information:
    st.text('\n')
    st.text('\n')
    st.text('\n')
    st.text('Sample preview of Triangles dataset.')
    img = Image.open('collages/EDA_triangles.png')
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


# Sidebar content for Predictor
elif choice == 'Predictor':

    st.subheader('Convolutional Neural Network - Realtime Predictor')
    
#Main Model = ('models/Final-MAIN-STARS_model_50_epochs_with_rescaler_zoom-0.2_rotation-45_skew-0.05_flip-hori_verti-02-main.h5') #Secondary Model = ('models/1-MAIN-STARS_model_50_epochs_with_rescaler_zoom-0.2_rotation-45_skew-0.03_flip-hori_verti-main.h5')   
#No-Stars Model = ('models/NO-STARS_model_with_rescaler_zoom-0.2_rotation-45_skew-0.05_flip-hori_verti-01.h5')
                 
                 
    #loading trained model for predicting
    model = tf.keras.models.load_model('models/Final-MAIN-STARS_model_50_epochs_with_rescaler_zoom-0.2_rotation-45_skew-0.05_flip-hori_verti-02-main.h5')
    #setup for drag and drop to run prediction model
    uploaded_file = st.file_uploader('Choose an image of a shape to classify...', type= ['png','jpg'])
    if uploaded_file is not None:
        #load image file
        im = Image.open(uploaded_file)        
        #display image and caption
        st.image(im, caption='Uploaded Image.',width=200, use_column_width=False)
        st.write("")
        st.write("Classifying at:")
        st.write("\n")
        #run inputted image through model
        predict(uploaded_file)

# Sidebar content for About
elif choice == 'About':

    st.subheader('About the Model')
    st.write("")
    st.write("")
                    
"""

##### Flatiron Final Project
- ###### Made by: Samuel Diaz
"""
