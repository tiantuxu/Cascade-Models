#!/usr/bin/env python3

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from train_lenet import LeNetLoader
from train_resnet34 import resnet34Loader
from train_resnet18 import resnet18Loader
from train_shallow_alexnet import AlexNetShallowLoader
from train_alexnet import AlexNetLoader
from time import time
import os
import copy
import argparse

class Test:
    def __init__(self, args):

        self.args = args
        self.outcomes = 10  # number of possible outcomes
        self.lr = 0.001  # learning rate
        self.epochs = 10  # default epoch
        self.batch = 100
        self.accuracy = 0
        # self.input_size = 28 * 28  # input image size

        # Inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py

        transform_test = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
            batch_size=1, shuffle=False, num_workers=4)

        # self.model = AlexNet()
        self.model1 = resnet18Loader(args.model1)
        self.model2 = resnet34Loader(args.model2)

        self.loss = nn.CrossEntropyLoss()
        self.valid_loss = 0
        self.threshold = args.thresh
        self.class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def validation(self):
        print('thresh ' + str(self.threshold))
        start = time()

        transform1 = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.model1.model.eval()
        self.model2.model.eval()
        correct = 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            onehot_target = torch.zeros(self.batch, self.outcomes)
            for n in range(len(target)):
                onehot_target[n][target[n]] = 1

            dataNew = copy.deepcopy(data)
            # data = transform2(data[0]).unsqueeze(0)
            # output = self.model2.model(data)
            # loss = self.loss(output, target)
            # _, index1 = torch.max(output, 1)
            # if index1 != target:
            #     print("idx: " + str(batch_idx) + " model2: " + str(index1) + " target: " + str(target))

            data1 = transform2(data[0]).unsqueeze(0)
            output = self.model1.model(data1)
            loss = self.loss(output, target)
            if float(loss.data) >= self.threshold:
                _, index1 = torch.max(output, 1)
                data2 = transform2(dataNew[0]).unsqueeze(0)
                output = self.model2.model(data2)
                loss = self.loss(output, target)
                # _, index2 = torch.max(output, 1)
                # print("idx: " + str(batch_idx) + " model1: " + str(index1) + " model2: " + str(index2) + " target: " + str(target))
            self.valid_loss += float(loss.data)

            _, index = torch.max(output, 1)
            for n in range(len(target)):
                if index[n] == target[n]:
                    correct += 1
            # if batch_idx > 0 and batch_idx % 100 == 0:
            #     print("progress: " + str(batch_idx / 100) + " / 100, accuracy = " + str(100.0 * correct / batch_idx))
        end = time()
        print("Time " + str(end - start) + " accuracy " + str(100.0 * correct / len(self.test_loader.dataset)))
        return 100.0 * correct / len(self.test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid Image Classifier Tester")
    parser.add_argument('--model1', type=str, required=True, help="Path for training data")
    parser.add_argument('--model2', type=str, required=True, help="Path for where trained model stored")
    parser.add_argument('--thresh', type=float, required=True, help="Path for where trained model stored")
    args = parser.parse_args()
    test = Test(args)
    # test.validation()
    for i in range(8, 3, -1):
        test.threshold = i * 0.2
        test.validation()
