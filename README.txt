## Overview
A Python feedforward neural network built with PyTorch that classifies 
fashion product images from the FashionMNIST dataset into 10 categories 
(e.g. dress, sneaker, trouser). Trained on 60,000 images and evaluated 
on 10,000 test images.

# Project Structure
   - classifier.py: Main Python script containing the neural network implementation, training and prediction functions.
   -log.txt: training log showing loss values and accuracy metrics during model training
   -Makefile: automation tool to run the classifier
   -README.txt: this file expalaining usage
   
# How to run
   make run
   
OR 
   
   python3 classifier.py
   
After training the program will prompt you to enter a filepath to an image classify:

   Please enter a filepath:
   > ./path/to/image.jpg
   
Enter exit to quit the program
