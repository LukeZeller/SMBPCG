from __future__ import print_function
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import common.level as lvl
from config import config_mgr
from gan.models import dcgan, mlp


GAN_DATA_REL = 'data/gan'
GAN_DATA_DIR_PATH = config_mgr.get_absolute_path(GAN_DATA_REL)
GAN_DATA_DIR = str(GAN_DATA_DIR_PATH)

DEFAULT_OUTPUT_FREQUENCY = 5000

# List of levels on which to train the GAN if --level isn't specified
# Note that SMB2 Japan levels 1-1 and 3-1 had heights of 12 and 15, respectively.
# The files were normalized to have height of 14 for consistency.
TRAINING_LEVELS = [
    'SuperMarioBros2(J)-World1-1_normalized.txt',
    'SuperMarioBros2(J)-World2-2.txt',
    'SuperMarioBros2(J)-World3-1_normalized.txt',
    'SuperMarioBros2(J)-World4-1.txt',
    'SuperMarioBros2(J)-World6-1.txt',
    'SuperMarioBros2(J)-World8-1.txt',
    'SuperMarioBros2(J)-WorldB-1.txt',
    'SuperMarioBros2(J)-WorldD-1.txt',
    'mario-1-1.txt',
    'mario-2-1.txt',
    'mario-3-1.txt',
    'mario-4-1.txt',
    'mario-5-1.txt',
    'mario-6-1.txt',
    'mario-7-1.txt',
    'mario-8-1.txt'
]

def _load_level(level_fname):
    level = lvl.load_level_from_ascii(level_fname)
    if level.height != lvl.DEFAULT_LEVEL_HEIGHT:
        raise ValueError("Level file must have height of 14.")
    return level
    

def _load_all_levels():
    levels = []
    for level_fname in TRAINING_LEVELS:
        levels.append(_load_level(level_fname))
    return levels

def _extract_segments_from_level(level, width = lvl.DEFAULT_LEVEL_WIDTH):
    n_segments = level.width - width + 1
    if n_segments <= 0:
        raise ValueError("Segment width cannot be greater than level width.")
    level_data = level.get_data(as_ndarray=True)
    segments = []
    for x_offset in range(n_segments):
        segments.append(level_data[ : , x_offset : x_offset + width])
    return segments

#Run with "python main.py"
def train(opt):
    print(opt)

    # Fix and setup seed for training
    if opt.seed == -1:
        seed = random.randint(1, 10000)
    else:
        seed = opt.seed
    print("[INFO] Using seed = {0}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)

    # Format and create output directory for training data
    if opt.output is None:
        opt.output = 'gan_training_output'
    output_dir = '{0}_{1}'.format(opt.output, seed)
    output_path = config_mgr.get_absolute_path(output_dir, GAN_DATA_DIR_PATH)

    os.system('mkdir {0}'.format(str(output_path)))

    if opt.save_frequency > 0:
        output_frequencey = opt.save_frequency
    else:
        output_frequency = DEFAULT_OUTPUT_FREQUENCY

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("[WARNING] You have a CUDA device, so you should probably run with --cuda")

    map_size = 32

    # Read all input files and organize into segments for training
    segments = []    
    if opt.level == '':
        levels = _load_all_levels()
    else:
        levels = [_load_level(opt.level)]
    for level in levels:
        segments.extend(_extract_segments_from_level(level))
    X = np.array(segments)

    print("[INFO] Beginning training on level data with shape = {0}".format(X.shape))

    z_dims = 10 # Number different title types

    num_batches = X.shape[0] / opt.batchSize

    X_onehot = np.eye(z_dims, dtype='uint8')[X]
    X_onehot = np.rollaxis(X_onehot, 3, 1)

    # Quick sanity check on dimensions before proceeding
    assert (X_onehot.shape[0] == X.shape[0] and
            X_onehot.shape[1] == z_dims and
            X_onehot.shape[2] == X.shape[1] and
            X_onehot.shape[3] == X.shape[2]
            )
    
    print("[INFO] Converted data to onehot representation with new shape = {0}".format(X_onehot.shape))

    # X_train and X_onehot have idental shape in the first two dimensions
    X_train = np.zeros ( (X.shape[0], z_dims, map_size, map_size) )

    X_train[:, 2, :, :] = 1.0  #Fill with empty space

    #Pad part of level so its a square
    X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    n_extra_layers = int(opt.n_extra_layers)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)

    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    input = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    def tiles2image(tiles):
        return plt.get_cmap('rainbow')(tiles/float(z_dims))

    def combine_images(generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[1:]
        image = np.zeros((height*shape[0], width*shape[1],shape[2]), dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
        return image

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
        print("Using ADAM")
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    gen_iterations = 0
    for epoch in range(opt.niter):
        #! data_iter = iter(dataloader)
        X_train = X_train[torch.randperm( len(X_train) )]
        
        i = 0
        while i < num_batches:#len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < num_batches:#len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = X_train[i*opt.batchSize:(i+1)*opt.batchSize]

                i += 1

                real_cpu = torch.FloatTensor(data)

                netD.zero_grad()
                #batch_size = num_samples #real_cpu.size(0)

                if opt.cuda:
                    real_cpu = real_cpu.cuda()

                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                errD_real = netD(inputv)
                errD_real.backward(one)

                # train with fake
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile = True) # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake
                errD_fake = netD(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[INFO] Training: [%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.niter, i, num_batches, gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if gen_iterations % output_frequency == 0:   #was 500
                
                fake = netG(Variable(fixed_noise, volatile=True))

                im = fake.data.cpu().numpy()
                #print('SHAPE fake',type(im), im.shape)
                #print('SUM ',np.sum( im, axis = 1) )

                im = combine_images( tiles2image( np.argmax( im, axis = 1) ) )

                plt.imsave('{0}/mario_fake_samples_{1}.png'.format(output_path, gen_iterations), im)
                torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(output_path, gen_iterations))
                if opt.saveD:
                    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(output_path, gen_iterations))

        # do checkpointing
        #torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        #torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
    
