# Shapes Image Classification - Using Convolutional Neural Networks



![](readme_images/wooden-baby-shape-puzzle-toy.jpg)

Flatiron Final Project - Shapes Image Classifier CNN

I wanted to make a image classifier using a convolutional neural network model. As well make system where I can feed a image and predict the correct class. Convolutional neural networks are one of many “deep learning” models; that just so happens works great with images and videos processing.

I made a Convolutional Neural Network to predict; a hand drawn shape, and classify if the image is one of four shapes (Circles, Squares, Stars or Triangles).

I also made a front-end drag and drop predictor app. The app will pass the hand drawn image through my model and will display the percentage the model thinks the shape is, Circle, Square, Star or Triangle.

## What Did I Do?

* Imported 2 image databases off Kaggle on shapes (Started with solid and outline shapes of circle, square and triangle)

* Manually scraped, cleaned and labeled 1000+ images to add stars as a shape (also added more hand drawn shapes).

* Did some basic Exploratory Data Analysis:

      All of the images in my dataset is in .png format.

      After splitting the dataset (4,480 total images) into Train/Test( 90/10 split ) of each shape:

      Total images in circles training set: 1013   --   Total images in circles test set: 114

      Total images in squares training set: 1013   --   Total images in squares test set: 113

      Total images in stars training set: 1002   --   Total  images in stars test set: 112

      Total images in triangles training set: 1008   --   Total images in triangles test set: 113

      I made a few collages (in photoshop) of a sampling of images, by their labels

    ![](readme_images/collage01.png)

    ![](readme_images/collage02.png)

* Built and tweaked my Neural Network.

* I then tuned my ImageDataGenerator, which allowed me to manipulate images to produce variations of an original image. This helped increase your data pool with the augmented version of original images.

      Samples of the ImageDataGenerator in action:
    ![](readme_images/ImageDataGenerator_example.png)   


* After training my model through 50 epochs and I calculated the average of the last ten epochs' of accuracy score (97.11%), average loss (0.084) and the average MSE (0.010). My model is performing very well as I have the classification report and the confusion matrix of my model's predictions

      Classification Report of model's performance:

    ![](readme_images/classification_report.png)

      Confusion Matrix of model's performance:

    ![](readme_images/confusion_matrix.png)

      Only 2 out of 448 possible outcomes were incorrectly classified (99.55% "accurate")

* Finally I made a Streamlit app(front-end) to live demo my model:

      Click image to see a video demoing my drag and drop predictor

    [![Alt text](https://i9.ytimg.com/vi/Y-tON5nfNnA/mq1.jpg?sqp=CK74_vEF&rs=AOn4CLBW9KAQTJ_HbthQ4yW1FTmo89FK2g)](https://www.youtube.com/watch?v=Y-tON5nfNnA&feature=emb_title)

      Just need to drag image of hand drawn shape into predictor and it will display the
      percentage the model thinks the shape is, whether a Circle, Square, Star or Triangle.







## Built With

* Python3.8
* Jupyter Notebook 6.0.0
* A few imports: pandas, numpy, tensorflow, keras, Pil and sklearn


## Authors

* **Samuel Diaz** - *Creator* - [sdman135](https://github.com/sdman135/)
