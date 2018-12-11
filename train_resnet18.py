#!/usr/bin/env python3

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import time
import os
import argparse
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import shutil
import numpy as np
# import cv2

# parser = argparse.ArgumentParser(description="Pretrained AlexNet for object classification")
# parser.add_argument('--data', type=str, help='path to training data')
# parser.add_argument('--save', type=str, help='path to saved model')
# args = parser.parse_args()

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

class resnet18Loader:
    def __init__(self, path):
        self.model = resnet18()
        self.model_name = 'Resnet18'
        self.accuracy = 0
        self.loss = nn.CrossEntropyLoss()
        self.load(path)
        print(self.model_name + " loaded from saved model, accuracy = " + str(self.accuracy))

    def load(self, path):
        #path = os.path.abspath(path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.accuracy = checkpoint['best_accuracy']

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class TrainModel:
    def __init__(self):
        '''
        def get_eval():
            path = os.path.join(args.data, 'val/images')
            filename = os.path.join(args.data, 'val/val_annotations.txt')
            f = open(filename, "r")
            data = f.readlines()

            val_img_dict = {}
            for line in data:
                words = line.split("\t")
                val_img_dict[words[0]] = words[1]
            f.close()

            for img, folder in val_img_dict.items():
                newpath = (os.path.join(path, folder))
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                if os.path.exists(os.path.join(path, img)):
                    os.rename(os.path.join(path, img), os.path.join(newpath, img))
        get_eval()
        def get_class(class_list):
            filename = os.path.join(args.data, 'words.txt')
            f = open(filename, "r")
            data = f.readlines()

            large_class_dict = {}
            for line in data:
                words = line.split("\t")
                super_label = words[1].split(",")
                large_class_dict[words[0]] = super_label[0].rstrip()
            f.close()

            tiny_class_dict = {}
            for small_label in class_list:
                for key, value in large_class_dict.items():
                    if small_label == key:
                        tiny_class_dict[key] = value
                        continue

            return tiny_class_dict
        '''

        '''500*200 images in tiby dataset'''
        self.train_batch_size = 100
        '''10000 images'''
        self.validation_batch_size = 10

        #train_path = os.path.join(args.data, 'train')
        #validation_path = os.path.join(args.data, 'val/images')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #train_data = datasets.ImageFolder(train_path, transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

        #validation_data = datasets.ImageFolder(validation_path, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
        transform_train = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomCrop(256, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #self.train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=5)
        self.train_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
            batch_size=self.train_batch_size, shuffle=True, num_workers=4)
        #self.validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=self.validation_batch_size, shuffle=False, num_workers=5)
        self.validation_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
            batch_size=self.validation_batch_size, shuffle=False, num_workers=4)

        #self.class_names = train_data.classes
        self.class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_classes = len(self.class_names)
        #self.tiny_class = get_class(self.class_names)

        '''Pretrained Model'''
        res = models.resnet18(pretrained=True)

        torch.manual_seed(1)
        self.model = resnet18()

        '''Transfer weights'''
        for i, j in zip(self.model.modules(), res.modules()):
            if not list(i.children()):
                if len(i.state_dict()) > 0:
                    if i.weight.size() == j.weight.size():
                        i.weight = j.weight
                        i.bias = j.bias

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.learning_rate = 0.0001
        self.epochs = 20
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)

        filename = 'checkpoint-resnet18.pth.tar'
        load_checkpoint_file = os.path.join(args.save, filename)
        if os.path.isfile(load_checkpoint_file):
            print ("Loading checkpoint file")
            checkpoint = torch.load(load_checkpoint_file)
            self.start_epoch = checkpoint['epoch']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.start_epoch = 0
            self.best_accuracy = 0

    def train(self):
        def save_checkpoint(state, better, file=os.path.join(args.save, 'checkpoint-resnet18.pth.tar')):
            torch.save(state, file)
            if better:
                shutil.copyfile(file, os.path.join(args.save, 'resnet18_model.pth.tar'))

        def training():
            self.model.train()
            training_loss = 0
            true_positive = 0

            for batch_id, (data, target) in enumerate(self.train_data_loader):
                data, target = Variable(data), Variable(target, requires_grad=False)
                self.optimizer.zero_grad()

                output = self.model(data)
                batch_loss = self.loss_fn(output, target)
                training_loss += batch_loss.data
                batch_loss.backward()
                self.optimizer.step()
                value, index = torch.max(output.data, 1)

                for i in range(0, self.train_batch_size):
                    if index[i] == target.data[i]:
                        true_positive += 1

            average_training_loss = training_loss / (len(self.train_data_loader.dataset) / self.train_batch_size)

            return float(100.0 * float(true_positive) / (len(self.train_data_loader.dataset))), average_training_loss

        def validation():
            self.model.eval()
            validation_loss = 0
            true_positive = 0

            for data, target in self.validation_data_loader:
                data, target = Variable(data), Variable(target, requires_grad=False)

                output = self.model(data)
                batch_loss = self.loss_fn(output, target)
                validation_loss += batch_loss.data
                value, index = torch.max(output.data, 1)

                for i in range(0, self.validation_batch_size):
                    if index[i] == target.data[i]:
                        true_positive += 1

            average_validation_loss = validation_loss / (len(self.validation_data_loader.dataset) / self.validation_batch_size)

            return 100.0 * float(true_positive) / (len(self.validation_data_loader.dataset)), average_validation_loss

        if self.epochs != self.start_epoch:
            print ("Starting training from epoch " + str(self.start_epoch + 1))
            for i in range(self.start_epoch + 1, self.epochs + 1):
                print ("--- Epoch " + str(i) + " ---")
                start = time.time()
                training_accuracy, training_loss = training()
                end = time.time()
                total_time = end - start
                validation_accuracy, validation_loss = validation()

                print ("Epoch = " + str(i) + " Training loss " + str(float(training_loss)) + " Training Accuracy " + str(float(training_accuracy)))
                print ("Epoch = " + str(i) + " Validation loss " + str(float(validation_loss)) + " Validation Accuracy " + str(validation_accuracy))
                print ("Total Training Time: " + str(total_time) + " s")

                better = validation_accuracy > self.best_accuracy
                self.best_accuracy = max(self.best_accuracy, validation_accuracy)
                print("Save model checkpoint after epoch " + str(i))
                save_checkpoint(
                    {'epoch': i,
                     'best_accuracy': self.best_accuracy,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'numeric_class_names': self.class_names,
                     'class_names': self.class_names,
                     }, better)
        else:
            print ("Training completed")

if __name__ == '__main__':
    res = TrainModel()
    res.train()
