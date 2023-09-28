# Importing the required libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard
import time
import csv
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import shutil
import argparse

# take the command line arguments
# 1. model name {default = vgg16}
# 2. blocks {default = 1}
# 3. epochs {default = 5}
# 4. batch size {default = 20}
# 5. image size {default = 128}
# 6. optimizer {default = adam}
# 7. learning rate {default = 0.001}

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model', type=str, default='vgg', help='model name')
parser.add_argument('--blocks', type=int, default=1, help='number of blocks 1,3,5,..')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--batch', type=int, default=5, help='batch size')
parser.add_argument('--imgsize', type=int, default=128, help='image size')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--augment', type=bool, default=False, help='data augmentation')
args = parser.parse_args()

assert args.model in ['vgg'], "Current model is not supported please try vgg"
# assert that the number of blocks is 1,3,16 else return that the number of blocks is not valid'
assert args.blocks in [1,3,16], "Current number of blocks is not supported please try 1,3,16"
assert args.epochs > 0, "Number of epochs should be greater than 0"
assert args.batch > 0, "Batch size should be greater than 0"
assert args.imgsize > 0, "Image size should be greater than 0"
assert args.optimizer in ['adam', 'sgd'], "Current optimizer is not supported please try adam or sgd"

print(args)

log_file = "results2.csv"
csv_header = ['Model Name', 'Training Time', 'Train Loss', 'Train Acc', 'Test Acc', 'Num Params', 'Optimizer', 'Learning Rate', 'Batch Size', 'Image Size', 'Epochs']
if not os.path.exists(log_file):
    with open(log_file, mode='a', newline='') as results_file:
        results_writer = csv.writer(results_file)
        results_writer.writerow(csv_header)


#In[]: __________________________ Define Parameters __________________________
# Define directories for training and testing data
train_dir = 'dataset/train/'
test_dir = 'dataset/test/'

# Define the image size to be used for resizing the images
img_size = (args.imgsize, args.imgsize)
input_img_size = (args.imgsize, args.imgsize, 3)
batch_size = args.batch
num_epochs = args.epochs


#In[]: __________________________ Define Data Generators __________________________
train_datagen, test_datagen = None, None
# Check if the model is VGG16
if args.model == 'vgg' and args.blocks == 16:
    # Define the preprocessing function for VGG16 model
    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    # Create data generator with the preprocessing function
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

elif args.augment:
    # Define the ImageDataGenerators for training and testing data with data augmentation and normalization
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                       rotation_range=40, 
                                       width_shift_range=0.2, 
                                       height_shift_range=0.2, 
                                       shear_range=0.2,
                                    #    zoom_range=0.2, 
                                    #    horizontal_flip=True, 
                                       fill_mode='nearest')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
else:
    # Define the ImageDataGenerators for training and testing data with data augmentation and normalization
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

# Create a generator for loading training data from the directory
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=img_size, # Resizes the images to a target size
                                                    batch_size=batch_size, # Defines the batch size
                                                    class_mode='categorical') # Defines the type of labels to use

# Create a generator for loading testing data from the directory
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=img_size, # Resizes the images to a target size
                                                  batch_size=batch_size, # Defines the batch size
                                                  class_mode='categorical') # Defines the type of labels to use
# Data generators for prediction
prediction_datagen = ImageDataGenerator(rescale=1./255)
preprocess_input = tf.keras.applications.vgg16.preprocess_input
prediction_datagen_vgg = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
prediction_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=1, class_mode='categorical', shuffle=False) 
prediction_generator_vgg = prediction_datagen_vgg.flow_from_directory(test_dir, target_size=img_size, batch_size=1, class_mode='categorical', shuffle=False) 



num_classes = len(train_generator.class_indices)

#In[]: __________________________ Define Model __________________________
log_dir = f'log_stats/{args.model}_{args.blocks}'

# VGG - 1 BLOCK
# Define a function that creates a VGG block with one convolutional layer
def vgg_1_block():
    # Create a Sequential model object with a name
    model = Sequential(name = 'vgg_block_1')
    # Add a convolutional layer with 64 filters, a 3x3 kernel size, 'relu' activation, and 'same' padding
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_img_size))
    model.add(MaxPooling2D((2, 2)))
    # Add a flatten layer to convert the 2D feature maps to a 1D feature vector
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))    
    model.add(Dense(num_classes, activation='softmax'))    #  output layer with num_classes unit and 'sigmoid' activation (for binary classification)
    return model

# VGG - 3 BLOCK
# Define a function to create a VGG block with three convolutional layers
def vgg_3_block():
    # Create a Sequential model object with the name 'vgg_block_3'
    model = Sequential(name='vgg_block_3')
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_img_size))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))    #  output layer with num_classes unit and 'sigmoid' activation (for binary classification)
    return model

def vgg_16_transfer_learning():
    # load model
    model = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=input_img_size)
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(num_classes, activation='softmax')(class1)
    # define new model
    model = keras.models.Model(inputs=model.inputs, outputs=output, name='vgg_16')
    return model

model1 = None
if args.model == 'vgg':
    if args.blocks == 16:
        # Transfer learning with VGG16
        model1 = vgg_16_transfer_learning()
    if args.blocks == 1:
        # Create a VGG block with one convolutional layer
        model1 = vgg_1_block()
    if args.blocks == 3:
        model1 = vgg_3_block()
    model1.summary() # Print a summary of the model's architecture
    # Compile the model with 'adam' optimizer, 'binary_crossentropy' loss function, and 'accuracy' metric
    model1.compile(optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    history = model1.fit(train_generator, steps_per_epoch=len(train_generator), epochs=num_epochs )
    training_time = time.time() - start_time

    # Evaluate the model 
    train_loss, train_acc = model1.evaluate(train_generator)
    test_loss, test_acc = model1.evaluate(test_generator)

    # Count the number of parameters in the model
    num_params = model1.count_params()

    # Open the results file in append mode and writing the results
    with open('results2.csv', mode='a', newline='') as results_file:
        results_writer = csv.writer(results_file)
        results_writer.writerow([f'{args.model}_{args.blocks}', training_time, train_loss, train_acc, test_acc, num_params, args.optimizer, args.lr, args.batch, args.imgsize, args.epochs])

    model1.save(f'models/{args.model}_{args.blocks}.h5')



