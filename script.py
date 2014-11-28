#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import argparse
from PIL import Image, ImageFilter

"""Ranks a tuple (r,g,b) between 0 and 16777215"""
def rank( (r, g, b) ):
    return r + g*256 + b*256*256

"""Returns a 20x20 window given the left and upper coordanates"""
def get_window(img, i, j): 
        box = (i, j, 20, 20) 
        return img.crop(box)

"""
Returns a vector of a linearly ranked (r,g,b) value of img
This function should be called with a 20x20 pixels img (window) 
"""
def get_features(img):
    (width, height) = img.size
    features = []
    for i in range(width):
        for j in range(height):
            v = rank(img.getpixel((i,j)))
            features.append(v)     # appending every pixel 

"""Given list of images (data_set), trains the network with backpropagation algorithm"""
def train_brain(data_set):
    for img in data_set:
        inputs = get_features(img)
        #call backprop inputs


"""Returns a list of images"""
def get_data_set(files):
    data_set = []
    for img in files:
        data_set.append(Image.open(img))
    return data_set

def main():
    parser = argparse.ArgumentParser(description='Face detection using Neural Networks')
    parser.add_argument('-t', '--train', help='Receives a list of images (training set)', nargs='+')
    parser.add_argument('-p', '--test', help='Receives a list of images (testing set)', nargs='+')
    parser.add_argument('-n', '--network', help='Read the file with the already trained network object', nargs=1)
    args = parser.parse_args()


if __name__ == "__main__":
        main()      
