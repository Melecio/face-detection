#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse

from random import shuffle

from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.supervised import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
# from pybrain.structure.modules import TanhLayer

from PIL import Image

import itertools

# Module image in 'image.py'
from image import img_features_vectors, img_features

"""Given list of images, train the network with the backpropagation algorithm"""
def train_data_set(files):
    # Because PyBrain may take the first 25% for testing
    shuffle(files)
    data_set = SupervisedDataSet(400, 1)
    for path, target in files:
        img = Image.open(path).convert('L')
        vector = img_features(img)
        img.close()
        data_set.addSample(vector, (target,))
    return data_set

"""Given list of images, test the network with the backpropagation algorithm"""
def test_brain(net, images):
    for img in images:
        for vector in img_features_vectors(img):
            pass
            # call backprop vector
        img.close()

"""Opens the images of the data set"""
def open_imgs(files):
    for path in files:
        yield Image.open(path).convert('L')

"""Parsing of the command-line input"""
def read():
    parser = argparse.ArgumentParser(description='Face detection using Neural Networks')
    parser.add_argument('-t', '--train-faces', help='Receives a list of images (training set)', nargs='+')
    parser.add_argument('-f', '--train-non-faces', help='Receives a list of images (training set)', nargs='+')
    parser.add_argument('-p', '--test', help='Receives a list of images (testing set)', nargs='+')
    parser.add_argument('-n', '--network', help='Read the file with the already trained network object', nargs=1)
    parser.add_argument('-w', '--write', help='Write the network to the specified file (format is .xml)', nargs=1)

    args = parser.parse_args()

    # Read the Neural Network Object
    if args.network != None:
        file_object = open(args.network, 'r')
        net = NetworkReader.readFrom(file_object)
    else:
        net = buildNetwork(400, 80, 16, 1, bias=True)
        # net = buildNetwork(400, 80, 16, 1, bias=True, hiddenclass=TanhLayer)

    # If there are some files to train with
    if (args.train_faces != None or args.train_non_faces != None):
        if args.train_faces != None:
            faces = args.train_faces
        else:
            faces = []

        if args.train_non_faces != None:
            non_faces = args.train_non_faces
        else:
            non_faces = []

        # Expected targets
        faces     = map(lambda path: (path, 1), faces)
        non_faces = map(lambda path: (path, 0), non_faces)

        training_files = faces + non_faces
    else:
        training_files = None

    # If there are some files to test with
    if args.test != None:
        testing_set = open_imgs(args.test)
    else:
        testing_set = None

    return net, training_files, testing_set, args.write

"""Main function"""
def main():
    net, training_files, testing_set, write_file = read()

    if training_files:
        training_set = train_data_set(training_files)
        trainer = BackpropTrainer(net, training_set, learningrate=0.01, verbose=True)
        trainer.train()

        if write_file:
            NetworkWriter.writeToFile(net, write_file)

if __name__ == "__main__":
    main()
