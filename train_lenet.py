#!/usr/bin/env python3

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from time import time
import os
import argparse


class AlexNet(nn.Module):
    def __init__(self):
        # Inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        # print(img.size())
        x = self.features(img)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # x = F.softmax(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNetLoader:
    def __init__(self, path):
        self.model = LeNet()
        self.model_name = 'LeNet'
        self.accuracy = 0
        self.loss = nn.CrossEntropyLoss()
        self.load(path)
        print(self.model_name + " loaded from saved model, accuracy = " + str(self.accuracy))

    def load(self, path):
        # loadpath = os.path.join(path, 'saved_model')
        load_model = torch.load(path)
        self.model.load_state_dict(load_model['state_dict'])
        # self.optimizer.load_state_dict(load_model['optimizer'])
        # self.class_names = load_model['class_names']
        # self.lr = load_model['lr']
        self.accuracy = load_model['accuracy']


class Img2obj:
    def __init__(self, args):

        self.args = args
        self.outcomes = 10  # number of possible outcomes
        self.lr = 0.001  # learning rate
        self.epochs = 10  # default epoch
        self.batch = 100
        self.accuracy = 0
        # self.input_size = 28 * 28  # input image size

        # Inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py

        transform_train = transforms.Compose([
            # transforms.Resize(224),
            # transforms.RandomCrop(256, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
            batch_size=self.batch, shuffle=True, num_workers=4)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
            batch_size=1, shuffle=False, num_workers=4)

        # self.model = AlexNet()
        self.model = LeNet()
        self.model_name = 'LeNet'
        # Copy weights
        # alexnet = models.alexnet(pretrained=True)
        #
        # for i, j in zip(self.model.modules(), alexnet.modules()):
        #     if not list(i.children()):
        #         if len(i.state_dict()) > 0:
        #             if i.weight.size() == j.weight.size():
        #                 i.weight.data = j.weight.data
        #                 i.bias.data = j.bias.data
        #
        # for params in self.model.parameters():
        #     params.requires_grad = False
        # for params in self.model.classifier[6].parameters():
        #     params.requires_grad = True
        #
        # self.optimizer = optim.Adam(self.model.classifier[6].parameters(), lr=lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.train_loss = 0
        self.valid_loss = 0

        self.class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def train(self):
        def train_epoch():
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)
                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.train_loss += float(loss.data)
                self.optimizer.step()

        def validation():
            self.model.eval()
            correct = 0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)
                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1
                output = self.model(data)
                loss = self.loss(output, target)
                self.valid_loss += float(loss.data)
                _, index = torch.max(output, 1)
                for n in range(len(target)):
                    if index[n] == target[n]:
                        correct += 1
                    #     print(loss.data)
                    # else:
                    #     print(str(loss.data) + ' +')


            return 100.0 * correct / len(self.test_loader.dataset)

        for i in range(self.epochs):
            self.train_loss = 0
            self.valid_loss = 0
            start = time()
            train_epoch()
            end = time()
            accuracy = validation()
            self.accuracy = accuracy
            end1 = time()

            print("Train epoch " + str(i+1) + ", train " + str(end - start) + ", valid " + str(end1 - end) +
                  " , accuracy " + str(accuracy) +
                  ", train loss " + str(self.train_loss / (len(self.train_loader.dataset) / self.batch)) +
                  ", valid loss " + str(self.valid_loss / (len(self.test_loader.dataset) / self.batch)))

    def forward(self, img):
        self.model.eval()
        img = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        output = self.model(img)
        _, index = torch.max(output, 1)
        return self.class_names[index.data[0]]

    def save(self, path):
        print("Saving models")
        save_model = {'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'class_names': self.class_names,
                     'accuracy': self.accuracy,
                     'lr': self.lr}
        print(os.path.join(path, self.model_name + "_" + str(self.accuracy)))
        torch.save(save_model, os.path.join(path, self.model_name + "_" + str(int(self.accuracy))))

    def load(self, path):
        path = os.path.join(os.path.abspath(path), 'saved_model')
        load_model = torch.load(path)
        self.model.load_state_dict(load_model['state_dict'])
        self.optimizer.load_state_dict(load_model['optimizer'])
        self.class_names = load_model['class_names']
        self.lr = load_model['lr']
        self.accuracy = load_model['accuracy']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def cam(self, idx=0):
        pass
        # cam = cv2.VideoCapture(idx)
        # font = cv2.FONT_HERSHEY_COMPLEX
        # cam.set(3, 640)
        # cam.set(4, 640)
        #
        # while True:
        #     read, frame = cap_obj.read()
        #
        #     if read:
        #         image_tensor = preprocess(frame)
        #
        #         predict = self.forward(image_tensor)
        #
        #         cv2.putText(frame, predict, (200, 80), font, 2, (173, 66, 244), 5, cv2.LINE_AA)
        #         cv2.imshow('Webcam Live Video', frame)  # Displaying the frame.
        #
        #     else:
        #         break
        #
        #     key_press = cv2.waitKey(1) & 0xFF
        #     if key_press == ord('q'):
        #         break
        #
        # cam.release()
        #
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    # torch.set_printoptions(threshold=5000)
    parser = argparse.ArgumentParser(description="AlexNet Image Classifier")
    parser.add_argument('--data', type=str, required=False, help="Path for training data")
    parser.add_argument('--save', type=str, required=False, help="Path for where trained model stored")
    args = parser.parse_args()
    img2obj = Img2obj(args)
    if args.data:
        img2obj.load(args.data)
        print("Load from saved model, accuracy = " + str(img2obj.accuracy))
    while True:
        prev = img2obj.accuracy
        img2obj.train()
        img2obj.save(args.save)
        img2obj.lr /= 10
        for param_group in img2obj.optimizer.param_groups:
            param_group['lr'] = img2obj.lr

