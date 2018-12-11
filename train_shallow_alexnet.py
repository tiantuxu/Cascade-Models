import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
import os, sys
import time
import numpy as np
import shutil
import argparse

# parser = argparse.ArgumentParser(description="Pretrained AlexNet for object classification")
# parser.add_argument('--data', type=str, help='path to training data')
# parser.add_argument('--save', type=str, help='path to saved model')
# args = parser.parse_args()

class AlexNetShallow(nn.Module):
    def __init__(self):
        super(AlexNetShallow, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )

    def forward(self, input):
        output = self.features(input)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        output = torch.nn.functional.softmax(output, dim = 1)
        return output


class AlexNetShallowLoader:
    def __init__(self, path):
        self.model = AlexNetShallow()
        self.model_name = 'AlexNetShallow'
        self.accuracy = 0
        self.loss = nn.CrossEntropyLoss()
        self.load(path)
        print(self.model_name + " loaded from saved model, accuracy = " + str(self.accuracy))

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.accuracy = checkpoint['best_accuracy']

class TrainModel:
    def __init__(self):
        '''500*200 images in tiby dataset'''
        self.train_batch_size = 100
        '''10000 images'''
        self.validation_batch_size = 10

        # train_path = os.path.join(args.data, 'train')
        # validation_path = os.path.join(args.data, 'val/images')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # train_data = datasets.ImageFolder(train_path, transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

        # validation_data = datasets.ImageFolder(validation_path, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
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
        # self.train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=5)
        self.train_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
            batch_size=self.train_batch_size, shuffle=True, num_workers=4)
        # self.validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=self.validation_batch_size, shuffle=False, num_workers=5)
        self.validation_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
            batch_size=self.validation_batch_size, shuffle=False, num_workers=4)

        # self.class_names = train_data.classes
        self.class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_classes = len(self.class_names)

        '''Pretrained Model'''
        #pretrained_alexnet = models.alexnet(pretrained=True)

        #torch.manual_seed(1)
        self.model = AlexNetShallow()

        '''Transfer weights'''
        '''
        for i, j in zip(self.model.modules(), pretrained_alexnet.modules()):
            if not list(i.children()):
                if len(i.state_dict()) > 0:
                    if i.weight.size() == j.weight.size():
                        i.weight.data = j.weight.data
                        i.bias.data = j.bias.data

        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.classifier[5].parameters():
            param.requires_grad = True
        '''
        self.learning_rate = 0.0001
        self.epochs = 20
        self.loss_fn = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(self.model.classifier[5].parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #self.train_loss = 0
        #self.valid_loss = 0

        self.class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        filename = 'checkpoint-alexnet-shallow.pth.tar'
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
        def save_checkpoint(state, better, file=os.path.join(args.save, 'checkpoint-alexnet-shallow.pth.tar')):
            torch.save(state, file)
            if better:
                shutil.copyfile(file, os.path.join(args.save, 'alexnet_model-shallow.pth.tar'))

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
    alex = TrainModel()
    alex.train()
