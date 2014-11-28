#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse

from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.supervised import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SigmoidLayer

from PIL import Image

import itertools

# Module image in 'image.py'
from image import img_features_vectors, img_features

"""Given list of images, trains the network with the backpropagation algorithm"""
def train_brain(net, data_set):
    for img, target in data_set:
        vector = img_features(img)
        img.close()
        # call backprop vector

"""Given list of images, test the network with the backpropagation algorithm"""
def test_brain(net, images):
    for img in images:
        for vector in img_features_vectors(img):
            pass
            # call backprop vector
        img.close()

"""Opens the images of the data set"""
def get_data_set(files, target):
    for path in files:
        yield (Image.open(path).convert('L'), target)

def main():
    parser = argparse.ArgumentParser(description='Face detection using Neural Networks')
    parser.add_argument('-t', '--train-faces', help='Receives a list of images (training set)', nargs='+')
    parser.add_argument('-f', '--train-non-faces', help='Receives a list of images (training set)', nargs='+')
    parser.add_argument('-p', '--test', help='Receives a list of images (testing set)', nargs='+')
    parser.add_argument('-n', '--network', help='Read the file with the already trained network object', nargs=1)

    args = parser.parse_args()

    # No need to train. Just read the Neural Network Object
    if (args.network != None):
        print "net"
        file_object = open(args.network, 'r')
        net = NetworkReader.readFrom(file_object)
    else:
        print "no net"
        net = buildNetwork(400, 20, 1, bias=True, hiddenclass=SigmoidLayer)

    # If there's something to train with
    if (args.train_faces != None or args.train_non_faces != None):
        print "train"
        trainer = BackpropTrainer(net, learningrate=0.01, verbose=True)

        if args.train_faces != None:
            faces = get_data_set(args.train_faces, 1)
        else:
            faces = []

        if args.train_non_faces != None:
            non_faces = get_data_set(args.train_non_faces, 0)
        else:
            non_faces = []

        training_set = itertools.chain(faces, non_faces)
        train_brain(net, training_set)

    if (args.test != None):
        print "test"
        testing_set = get_data_set(args.test)
        test_brain(net, testing_set)

if __name__ == "__main__":
    main()
