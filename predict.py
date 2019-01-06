import argparse
parser = argparse.ArgumentParser(
    description='This is an app to train a neural network on the input data')

parser.add_argument('path_to_image', action='store',
                    help='store the path to image')
parser.add_argument('checkpoint', action='store',
                    help='store the checkpoint')
parser.add_argument('-top', '--topk', default=5, type=int)
parser.add_argument('-catna', '--cattoname', default='cat_to_name.json')
parser.add_argument('-g', '--gpu', default='cpu')
args = parser.parse_args()
device=args.gpu
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import pandas as pd
from PIL import Image

from functions import load_checkpoint
from functions import process_image
from functions import imshow
from functions import predict

import json

cattoname=args.cattoname
with open(cattoname, 'r') as f:
    cat_to_name = json.load(f)
    
model, classtoindex = load_checkpoint(args.checkpoint)

image_path=args.path_to_image
Image.open(image_path)

probability, df= predict(image_path, model, classtoindex, args.topk, device)
df['name']= df.classes.map(cat_to_name)
probability=probability.cpu()
probability= probability.detach().numpy()
df['probability']=np.ravel(probability)
#df['probability']=np.ravel(probability)
print (df)

