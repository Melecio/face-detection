#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from os import listdir
from os.path import isfile, join

from random import shuffle

from pybrain.datasets                import ClassificationDataSet
from pybrain.tools.shortcuts         import buildNetwork
from pybrain.supervised.trainers     import BackpropTrainer
from pybrain.structure.modules       import SoftmaxLayer, TanhLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

from PIL import Image

import itertools

# Module image in 'image.py'
from image import img_features_vectors, img_features

"""Given list of images, create the training data set"""
def train_data_set(files):
    # Because PyBrain may take the first 25% for testing
    # shuffle(files)
    data_set = ClassificationDataSet(400, nb_classes=2, class_labels=['face', 'non-face'])
    for path, target in files:
        img = Image.open(path).convert('L')
        vector = img_features(img)
        img.close()
        data_set.addSample(vector, target)
    return data_set

"""Given list of images, test the network with the backpropagation algorithm"""
def test_network(net, images):
    for img in images:
        for vector in img_features_vectors(img):
            print net.activate(vector)
        img.close()

"""Opens the images of the data set"""
def open_imgs(files):
    for path in files:
        yield Image.open(path).convert('L')

"""Given a directory, opens it and gets the files"""
def get_files(directory):
    files = listdir(directory)
    paths = map(lambda f: join(directory,f), files)
    return [ p for p in paths if isfile(p) ]

"""Parsing of the command-line input"""
def read():
    parser = argparse.ArgumentParser(description='Face detection using Neural Networks')
    parser.add_argument('-t', '--train-faces', help='Receives a directory with files to train with', nargs='+')
    parser.add_argument('-f', '--train-non-faces', help='Receives a directory with files to train with', nargs='+')
    parser.add_argument('-p', '--test', help='Receives a list of images (testing set)', nargs='+')
    parser.add_argument('-r', '--read', help='Read the file with the already trained network object', nargs=1)
    parser.add_argument('-w', '--write', help='Write the network to the specified file (format is .xml)', nargs=1)

    args = parser.parse_args()

    # Read the Neural Network Object
    if args.read:
        net = NetworkReader.readFrom(args.read[0])
    else:
        net = buildNetwork(400, 20, 1, bias=True, outclass=SoftmaxLayer)
        # net = buildNetwork(400, 80, 16, 1, bias=True, hiddenclass=TanhLayer)

    # If there are some files to train with
    if (args.train_faces or args.train_non_faces):

        if args.train_faces:
            faces = get_files(args.train_faces[0])
        else:
            faces = []

        if args.train_non_faces:
            non_faces = get_files(args.train_non_faces[0])
        else:
            non_faces = []

        # Expected targets
        faces     = map(lambda path: (path, (1,)), faces)
        non_faces = map(lambda path: (path, (0,)), non_faces)

        training_files = faces + non_faces
    else:
        training_files = None

    # If there are some files to test with
    if args.test:
        testing_imgs = open_imgs(args.test)
    else:
        testing_imgs = None

    # If there is a writing file
    if args.write:
        write_file = args.write[0]
    else:
        write_file = None

    return net, training_files, testing_imgs, write_file

"""Main function"""
def main():
    net, training_files, testing_imgs, write_file = read()

    if training_files:
        print "creating training data set"
        training_set = train_data_set(training_files)
        print "training"
        print training_set, len(training_set)
        training_set.saveToFile('train.set')
        trainer = BackpropTrainer(net, training_set)
        trainer.trainEpochs(1)

    if testing_imgs:
        print "testing"
        test_network(net, testing_imgs)

    if write_file:
        NetworkWriter.writeToFile(net, write_file)

if __name__ == "__main__":
    main()
