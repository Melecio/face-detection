#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
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
            

def main():
    img = Image.open(sys.argv[1])
    img.load()

if __name__ == "__main__":
        main()      
