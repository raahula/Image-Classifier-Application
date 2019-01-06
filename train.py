import argparse
parser = argparse.ArgumentParser(
    description='This is an app to train a neural network on the input data')

parser.add_argument('data_directory', action='store',
                    help='store the data directory address')
parser.add_argument('-sd', '--save_dir', default='checkpoint.pth')
parser.add_argument('-ar', '--arch', default='vgg16')
parser.add_argument('-l', '--learning_rate', default=0.001, type=float)
parser.add_argument('-hi', '--hidden_units', default=10000, type=int)
parser.add_argument('-e', '--epochs', default=1, type=int)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

# Imports here

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import pandas as pd
from functions import validation
from PIL import Image
from collections import OrderedDict


data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)


# TODO: Build and train your network

model = getattr(models, args.arch)(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

arch=args.arch
if arch[0:3]=='vgg':
    inputunits=25088
else:
    inputunits=1024

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputunits, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

# TODO: Train a model with a pre-trained network

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
print_every = 40
steps = 0

# change to cuda

if args.gpu:
    device='cuda'
else:
    device='cpu'

model.to(device)

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_dataloader):
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
           
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_dataloader, criterion, optimizer)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_dataloader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloader)))
            running_loss = 0  
            
            
            # Make sure training is back on
            model.train()

# TODO: Save the checkpoint 
optimizer_state= optimizer.state_dict

model.class_to_idx = train_dataset.class_to_idx
checkpoint = {'input_size': inputunits,
              'output_size': 102,
              'hidden_layers': args.hidden_units,
              'state_dict': model.state_dict(),
              'epochs': args.epochs,
              'class to index': model.class_to_idx,
              }
directory=args.save_dir
torch.save(checkpoint, directory)

