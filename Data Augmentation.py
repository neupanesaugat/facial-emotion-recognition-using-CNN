import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Define paths
test_dir = 'C:/Users/sauga/Documents/Minor Project (Code)/Augmented Train/test'
train_dir = 'C:/Users/sauga/Documents/Minor Project (Code)/Augmented Train/train'
save_dir = 'C:/Users/sauga/Documents/Minor Project (Code)/Augmented Train/train'

# Define the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

emotions = ['angry', 'happy', 'neutral', 'sad']

for emotion in emotions:
    # Create a list to store images
    images = []
    
    # Load images from test directory
    for filename in os.listdir(os.path.join(test_dir, emotion)):
        img = cv2.imread(os.path.join(test_dir, emotion, filename))
        if img is not None:
            images.append(img)
    
    # Load images from train directory
    for filename in os.listdir(os.path.join(train_dir, emotion)):
        img = cv2.imread(os.path.join(train_dir, emotion, filename))
        if img is not None:
            images.append(img)
    
    # Convert list of images to numpy array
    images = np.array(images)
    
    # Calculate the number of augmentation needed
    num_augment = 17978 - len(images)
    
    # Create save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create emotion directory inside save directory if it does not exist
    emotion_dir = os.path.join(save_dir, emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)
    
    # Perform augmentation and save the images
    i = 0
    for batch in datagen.flow(images, batch_size=1, save_to_dir=emotion_dir, save_prefix=emotion, save_format='jpg'):
        i += 1
        if i > num_augment:
            break


print("Augmentation complete.")







