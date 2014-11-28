#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse

from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

from PIL import Image

# Module image in 'image.py'
from image import img_features_vectors, img_features

"""Given list of images, trains the network with the backpropagation algorithm"""
def train_brain(net, images):
    for img in images:
        vector = img_features(img)
        print vector
        # call backprop vector

"""Given list of images, test the network with the backpropagation algorithm"""
def test_brain(net, images):
    for img in images:
        for vector in img_features_vectors(img):
            print vector
            # call backprop vector

"""Opens the images of the data set"""
def get_data_set(files):
    for img in files:
        yield Image.open(img).mode('L')

def main():
    parser = argparse.ArgumentParser(description='Face detection using Neural Networks')
    parser.add_argument('-t', '--train', help='Receives a list of images (training set)', nargs='+')
    parser.add_argument('-p', '--test', help='Receives a list of images (testing set)', nargs='+')
    parser.add_argument('-n', '--network', help='Read the file with the already trained network object', nargs=1)

    args = parser.parse_args()

    # No need to train. Just read the Neura Network Object
    if (args.network != None):
        print "net"
        file_object = open(args.network, 'r')
        net = NetworkReader.readFrom(file_object)
    else:
        print "no net"
        net = buildNetwork(400, 20, 1)

    if (args.train != None):
        print "train"
        training_set = get_data_set(args.train)
        train_brain(net, training_set)

    if (args.test != None):
        print "test"
        testing_set = get_data_set(args.test)
        test_brain(net, testing_set)

if __name__ == "__main__":
    main()
