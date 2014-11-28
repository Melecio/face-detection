#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from math import ceil

"""Ranks a pixel of an img, either in RGB, in L or in LA mode"""
def pixel_rank(img, pixel):
    if img.mode == 'RGB':
        (r,g,b) = img.getpixel(pixel)
        return int(round((r + g + b) / 3))
    elif img.mode == 'L':
        return img.getpixel(pixel)
    elif img.mode == 'LA':
        l, _ = img.getpixel(pixel)
        return l

"""Returns a 20x20 window given the left and top coordinates"""
def img_window(img, i, j):
    width, height = img.size

    if width < i + 20:
        i = width - 20
    if height < j + 20:
        j = height - 20

    box = (i, j, i+20, j+20)
    return img.crop(box)


"""
Returns a vector of a linearly ranked value of img
This function should be called with a 20x20 pixels img (window)
"""
def img_features(img):
    width, height = img.size
    vector = []
    for i in range(width):
        for j in range(height):
            ranked = pixel_rank(img, (i,j))
            vector.append(ranked)
    return vector

"""
Crop an image in 20x20 squares and yield each result
"""
def img_crops(img):
    width, height = img.size
    top = 0

    # Padding for taking every pixel into account
    if (width % 5) != 0:
        width += 5 - (width % 5) - 20
    else:
        width -= 20

    if (height % 5) != 0:
        height += 5 - (height % 5) - 20
    else:
        height -= 20

    while top <= height:
        left = 0
        while left <= width:
            yield img_window(img, left, top)
            left += 5
        top += 5

"""
Get all the windows of a given image to pass to the neural network
"""
def img_windows(img):
    width, height = img.size

    if width < 20 or height < 20:
        factor = 1
        factor = max(factor, 20.0 / width)
        factor = max(factor, 20.0 / height)

        width = int(ceil(width * factor))
        height = int(ceil(height * factor))

        img = img.resize((width, height), Image.ANTIALIAS)

    while width >= 20 and height >= 20:
        for crop in img_crops(img):
            yield crop

        width  = int(round(width / 1.2))
        height = int(round(height / 1.2))

        img = img.resize((width, height), Image.ANTIALIAS)

"""
Get all the representing vectors of the windows of a given image
"""
def img_features_vectors(img):
    for window in img_windows(img):
        yield img_features(window)
