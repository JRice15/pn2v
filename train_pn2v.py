import os

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

from dival.datasets.lodopab_dataset import LoDoPaBDataset
import dival

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--obsmodel',default='pre-log',help='observation model: post-log or pre-log')
parser.add_argument('--mode',default='samplepoisson',help='noise model: uncalib, poisson, fixedpoisson')
parser.add_argument('--reg',type=float,default=10,help='regularization weight on prior std. dev.')
parser.add_argument('--crop',type=int,default=128,help='crop size')
parser.add_argument('--batch',type=int,default=4,help='batch size')
parser.add_argument('--epoch',type=int,default=300,help='num epochs')
parser.add_argument('--steps',type=int,default=50,help='steps per epoch')
parser.add_argument('--lr',type=float,default=0.0003,help='learning rate')

args = parser.parse_args()

if args.mode != "samplepoisson":
    raise ValueError("Only support samplepoisson mode")


""" Load training and validation datasets """

# update dival config file with correct path to data
dival.config.set_config('/lodopab_dataset/data_path', args.path)

dataset = LoDoPaBDataset(observation_model=args.obsmodel, impl='skimage')

#sinogram, _ = dataset.get_sample(0, part='train', out=(True,False))
#height, width = sinogram.shape[0:2]
#y_crop = 32*(height//32)
#x_crop = 32*(width//32)
#print('crop: ',y_crop,x_crop)
#crop_size = [y_crop,x_crop]
crop_size = [args.crop,args.crop]

def load_train_sinograms(batch_size, crop_size):
    indices = np.arange(dataset.train_len) 
    batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))

    while True:
        np.random.shuffle(indices)

        for batch_idx, file_idx in enumerate(indices):
            sinogram, _ = dataset.get_sample(file_idx, part='train', out=(True,False))
            if args.mode == 'uncalib':
              sinogram = np.copy(sinogram)*2-1
            elif args.mode == 'samplepoisson':
              sinogram *= 1000

            y = np.random.randint(sinogram.shape[0]-crop_size[0])
            x = np.random.randint(sinogram.shape[1]-crop_size[1])
            batch[batch_idx%batch_size] = sinogram[y:y+crop_size[0],x:x+crop_size[1],None]

            if (batch_idx+1) % batch_size == 0:
                yield batch, None 

train_generator = load_train_sinograms(args.batch, crop_size)

def load_validation_sinograms(num_samples, crop_size):
    val_crops = []
    indices = np.random.choice(dataset.validation_len, num_samples, replace=False) 
    #indices = trange(dataset.validation_len)

    for idx in indices:
        sinogram, _ = dataset.get_sample(idx, part='validation', out=(True,False))
        if args.mode == 'uncalib':
          sinogram = np.copy(sinogram)*2-1
        elif args.mode == 'samplepoisson':
          sinogram *= 1000

        y = np.random.randint(sinogram.shape[0]-crop_size[0])
        x = np.random.randint(sinogram.shape[1]-crop_size[1])
        val_crops.append(sinogram[y:y+crop_size[0],x:x+crop_size[1],None])

    return np.stack(val_crops, axis=0)

val_sinograms = load_validation_sinograms(100, crop_size)
print('Validation set size: %d'%len(val_sinograms))
print('Validation shape: %s'%(val_sinograms.shape,))



# %% [markdown]
# ### Create the Network and Train it
# This can take a while.

# %%
# Create a network with 800 output channels that are interpreted as samples from the prior.
net = UNet(800, depth=3)

# Split training and validation data.
# my_train_data=data[:-5].copy()
# my_val_data=data[-5:].copy()

# Start training.
os.makedirs("output", exist_ok=True)
trainHist, valHist = training.trainNetwork(net=net, trainData=train_generator, valData=val_sinograms,
                                           postfix='conv', directory="output", 
                                           #noiseModel=noiseModel,
                                           patchSize=args.crop,
                                           device=device, numOfEpochs= 200, 
                                           stepsPerEpoch=5, virtualBatchSize=20,
                                           batchSize=1, learningRate=1e-3)


# %%
# Let's look at the training and validation loss
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(valHist, label='validation loss')
plt.plot(trainHist, label='training loss')
plt.legend()
plt.savefig("output/loss.png")


# %%



