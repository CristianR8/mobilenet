import os
import cv2
import json
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from skimage.transform import resize

class DataGen:

    def __init__(self, path, split_ratio, x, y, color_space='rgb'):
        self.x = x
        self.y = y
        self.path = path
        self.color_space = color_space
        self.path_train_images = path 
        self.path_train_labels = path + "mask/"
        self.image_file_list = get_image_list(self.path_train_images)
        self.label_file_list = get_image_list(self.path_train_labels)
        self.image_file_list[:], self.label_file_list[:] = self.shuffle_image_label_lists_together()
        self.split_index = int(split_ratio * len(self.image_file_list))
        self.x_train_file_list = self.image_file_list[self.split_index:]
        self.y_train_file_list = self.label_file_list[self.split_index:]
        self.x_test_file_list = self.image_file_list[:self.split_index]
        self.y_test_file_list = self.label_file_list[:self.split_index]

    def generate_data(self, train=False, test=False):
        """Replaces Keras' native ImageDataGenerator."""
        try:
            if train is True:
                image_file_list = self.x_train_file_list
                label_file_list = self.y_train_file_list
            elif test is True:
                image_file_list = self.x_test_file_list
                label_file_list = self.y_test_file_list
        except ValueError:
            print('one of train or val or test need to be True')

        width_shape, height_shape = self.x, self.y # Tamaño de las imagenes de entrada
        imagenes = np.array(list(map(lambda x: resize(plt.imread(self.path_train_images + x), (height_shape, width_shape),mode='constant', preserve_range=True), tqdm(image_file_list))))
        imagenes = imagenes.astype('float32') 
        
        labels = []
        for label in tqdm(label_file_list):
            label = plt.imread(self.path_train_labels + label)
            label = resize(label, (height_shape, width_shape),mode='constant', preserve_range=True)
            label = np.expand_dims(label, axis=2)
            labels.append(label.astype('float32'))

        imagenes = normalize(np.array(imagenes))
        labels = normalize(np.array(labels))

        return imagenes,labels


    def get_num_data_points(self, train=False, val=False):
        try:
            image_file_list = self.x_train_file_list if val is False and train is True else self.x_val_file_list
        except ValueError:
            print('one of train or val need to be True')

        return len(image_file_list)

    def shuffle_image_label_lists_together(self):
        combined = list(zip(self.image_file_list, self.label_file_list))
        random.shuffle(combined)
        return zip(*combined)

    @staticmethod
    def change_color_space(image, label, color_space):
        if color_space.lower() == 'hsi' or 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
        elif color_space.lower() == 'lab':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)
        return image, label
def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


def get_image_list(path):
    file_list = [item for item in os.listdir(path) if os.path.isfile(os.path.join(path, item))]
    file_list.sort()
    return file_list

