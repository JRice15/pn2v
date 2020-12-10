# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # PN2V Prediction
# Please run the 'Convallaria-1-CreateNoiseModel' and 'Convallaria-2-Training' notebooks first.

import argparse
import os

import dival
from dival.util.constants import MU_MAX
from dival.datasets.lodopab_dataset import LoDoPaBDataset
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from imageio import imwrite

import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import imread

import pn2v.training
from pn2v import histNoiseModel, prediction, utils
from pn2v.utils import PSNR, denormalize, normalize
from unet.model import UNet

# See if we can use a GPU
device=utils.getDevice()

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--crop',type=int,default=128,help='crop size for training data')
parser.add_argument('--obsmodel',default='pre-log',help='observation model: post-log or pre-log')
parser.add_argument('--mode',default='uncalib',help='noise model: mse, uncalib, gaussian, poisson, poissongaussian')
parser.add_argument('--reg',type=float,default=10,help='regularization weight on prior std. dev.')

args = parser.parse_args()

# %% [markdown]
# ### Load Data

# %%
# update dival config file with correct path to data
dival.config.set_config('/lodopab_dataset/data_path', args.path)

dataset = LoDoPaBDataset(observation_model=args.obsmodel, impl='astra_cpu')
fbp = FBPReconstructor(dataset.get_ray_trafo(impl='astra_cpu'))

# compute padded (height, width) for sinograms to ensure they are multiples of 32
height, width = dataset.shape[0]
input_height = (height // 32 + 1) * 32 + 256
input_width = (width // 32 + 1) * 32 + 256
#input_height = (height // args.crop + 1) * args.crop
#input_width = (width // args.crop + 1) * args.crop
#pad_bottom = input_height - height
#pad_right = input_width - width
hdiff = input_height - height
wdiff = input_width - width
pad_top = hdiff // 2
pad_left = wdiff // 2
pad_bottom = hdiff - hdiff // 2
pad_right = wdiff - wdiff // 2
minval = 0.1 / 4096

def load_sinograms():
    gen = dataset.generator(part='test')
    ray_transform = dataset.get_ray_trafo(impl='astra_cpu')

    for noisy_sinogram, gt_reconstruction in gen:
        # pad sinogram to be a multiple of 32
        #noisy_sinogram = np.pad(noisy_sinogram, ((0, pad_bottom), (0, pad_right)), 'constant')
        noisy_sinogram = np.pad(noisy_sinogram, ((pad_top, pad_bottom), (pad_left, pad_right)), 'symmetric')
        #imwrite('noisy_sinogram.tiff',noisy_sinogram)
        #sys.exit(0)

        # create ground truth sinogram from ground truth reconstructed image
        if args.obsmodel == 'pre-log':
            gt_sinogram = np.exp(-ray_transform(gt_reconstruction*MU_MAX))
        else:
            gt_sinogram = ray_transform(gt_reconstruction) 

        yield noisy_sinogram, gt_sinogram

test_generator = load_sinograms()

# We estimate the ground truth by averaging.
# dataTestGT=np.mean(dataTest[:,...],axis=0)[np.newaxis,...]

# %% [markdown]
# ### Load the Network and Noise Model

# %%
# We are loading the histogram from the 'Convallaria-1-CreateNoiseModel' notebook.
# histogram=np.load(path+'noiseModel.npy')

# Create a NoiseModel object from the histogram.
# noiseModel=histNoiseModel.NoiseModel(histogram, device=device)


# %%
# Load the network, created in the 'Convallaria-2-Training' notebook
net=torch.load("output/last_conv.net")

# %% [markdown]
# ### Evaluation

# %%
# Now we are processing data and calculating PSNR values.
results=[]
meanRes=[]
resultImgs=[]
inputImgs=[]

results_path = "output/ims/"
os.makedirs(results_path, exist_ok=True)

# We iterate over all test images.
for index,(im,gt) in enumerate(test_generator):
    
    # im=dataTest[index]
    # gt=dataTestGT[0] # The ground truth is the same for all images
    
    # We are using tiling to fit the image into memory
    # If you get an error try a smaller patch size (ps)
    means, mseEst = prediction.tiledPredict(im*1000, net,
                                            ps=256, overlap=48,
                                            device=device,
                                            noiseModel="this is a non-None placeholder"
                                        )
    
    resultImgs.append(mseEst)
    inputImgs.append(im)

    rangePSNR=np.max(gt)-np.min(gt)
    psnr=PSNR(gt, mseEst,rangePSNR )
    psnrPrior=PSNR(gt, means,rangePSNR )
    results.append(psnr)
    meanRes.append(psnrPrior)

    print ("image:",index)
    print ("PSNR input",PSNR(gt, im, rangePSNR))
    print ("PSNR prior",psnrPrior) # Without info from masked pixel
    print ("PSNR mse",psnr) # MMSE estimate using the masked pixel
    print ('-----------------------------------')

    noisy_sinogram = np.squeeze(noisy_sinogram)#*maxval

    imwrite(os.path.join(results_path,'%04d_noisy_prelog_sinogram.tif'%index),noisy_sinogram)
    imwrite(os.path.join(results_path,'%04d_denoised_prelog_sinogram.tif'%index),denoised_sinogram)

    # perform log-transform if necessary     
    if args.obsmodel == 'pre-log':
        noisy_sinogram = -np.log(noisy_sinogram) / MU_MAX
        denoised_sinogram = -np.log(denoised_sinogram) / MU_MAX
        gt_sinogram = -np.log(gt_sinogram) / MU_MAX

    imwrite(os.path.join(results_path,'%04d_noisy_postlog_sinogram.tif'%index),noisy_sinogram)
    imwrite(os.path.join(results_path,'%04d_denoised_postlog_sinogram.tif'%index),denoised_sinogram)

    # reconstruct image from denoised sinogram
    noisy_reconstruction = fbp.reconstruct(noisy_sinogram)
    denoised_reconstruction = fbp.reconstruct(denoised_sinogram)
    
    imwrite(os.path.join(results_path,'%04d_noisy_reconstruction.tif'%index),noisy_reconstruction)
    imwrite(os.path.join(results_path,'%04d_denoised_reconstruction.tif'%index),denoised_reconstruction)
    
    
# We display the results for the last test image
vmi=np.percentile(gt,0.01)
vma=np.percentile(gt,99)

plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.title(label='Input Image')
plt.imshow(im, vmax=vma, vmin=vmi, cmap='magma')

plt.subplot(1, 3, 2)
plt.title(label='Avg. Prior')
plt.imshow(means, vmax=vma, vmin=vmi, cmap='magma')

plt.subplot(1, 3, 3)
plt.title(label='PN2V-MMSE estimate')
plt.imshow(mseEst, vmax=vma, vmin=vmi, cmap='magma')
plt.show()

plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.title(label='Input Image')
plt.imshow(im[100:200,150:250], vmax=vma, vmin=vmi, cmap='magma')

plt.subplot(1, 3, 2)
plt.title(label='Avg. Prior')
plt.imshow(means[100:200,150:250], vmax=vma, vmin=vmi, cmap='magma')

plt.subplot(1, 3, 3)
plt.title(label='PN2V-MMSE estimate')
plt.imshow(mseEst[100:200,150:250], vmax=vma, vmin=vmi, cmap='magma')
plt.savefig("output/test_ims.png")

print("Avg PSNR Prior:", np.mean(np.array(meanRes) ), '+-(2SEM)',2*np.std(np.array(meanRes) )/np.sqrt(float(len(meanRes)) ) )
print("Avg PSNR MMSE:", np.mean(np.array(results) ),  '+-(2SEM)' ,2*np.std(np.array(results) )/np.sqrt(float(len(results)) ) )


