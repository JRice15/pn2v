# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # PN2V Training
# Here we will use the generated noise model and train a PN2V network on single noisy images.
# Please run the 'Convallaria-1-CreateNoiseModel' notebook first.

# %%
import matplotlib.pyplot as plt
import numpy as np
from unet.model import UNet
import argparse

from pn2v import utils
from pn2v import histNoiseModel
from pn2v import training
from tifffile import imread
# See if we can use a GPU
device=utils.getDevice()

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--obsmodel',default='pre-log',help='observation model: post-log or pre-log')
parser.add_argument('--mode',default='uncalib',help='noise model: uncalib, poisson, fixedpoisson')
parser.add_argument('--reg',type=float,default=10,help='regularization weight on prior std. dev.')
parser.add_argument('--crop',type=int,default=128,help='crop size')
parser.add_argument('--batch',type=int,default=4,help='batch size')
parser.add_argument('--epoch',type=int,default=300,help='num epochs')
parser.add_argument('--steps',type=int,default=50,help='steps per epoch')
parser.add_argument('--lr',type=float,default=0.0003,help='learning rate')

args = parser.parse_args()

# %% [markdown]
# ### Load Data

# %%
path='data/Convallaria_diaphragm/'

# Load the training data
data=imread(path+'20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif')


# %%
# We are loading the histogram from the 'Convallaria-1-CreateNoiseModel' notebook
histogram=np.load(path+'noiseModel.npy')

# Create a NoiseModel object from the histogram.
noiseModel=histNoiseModel.NoiseModel(histogram, device=device)

# %% [markdown]
# ### Create the Network and Train it
# This can take a while.

# %%
# Create a network with 800 output channels that are interpreted as samples from the prior.
net = UNet(800, depth=3)

# Split training and validation data.
my_train_data=data[:-5].copy()
my_val_data=data[-5:].copy()

# Start training.
trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                           postfix='conv', directory=path, noiseModel=noiseModel,
                                           device=device, numOfEpochs= 200, stepsPerEpoch=5, virtualBatchSize=20,
                                           batchSize=1, learningRate=1e-3)


# %%
# Let's look at the training and validation loss
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(valHist, label='validation loss')
plt.plot(trainHist, label='training loss')
plt.legend()
plt.show()


# %%



