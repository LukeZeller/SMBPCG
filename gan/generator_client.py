# Module for using trained generator to transform latent vectors into levels
# Code in this file is based heavily on generator_ws.py from DagstuhlGAN

import numpy as np
import torch
# TODO: Refactor out Variable as it is depracated
from torch.autograd import Variable

from config import config_mgr
from common.level import Level
import gan.models.dcgan as dcgan

# ---- Default model parameters for the generator ----
DEF_GEN_MODEL_FILE = "netG_epoch_10000_0_32.pth"
DEF_LV_LEN = 32
DEF_BATCH_SZ = 1
DEF_IMAGE_SZ = 32
DEF_NGF = 64
DEF_NGPU = 1
DEF_EXTRA_LAYERS = 0

# Number of different tiles (dimension of output vectors)
NUM_TILES = 10

# TODO: Refactor into Generator proxy class for cleanliness
# TODO: Clean this clusterfuck of code up
generator = None


def load_generator(model_file=DEF_GEN_MODEL_FILE,
                   nz=DEF_LV_LEN,
                   image_sz=DEF_IMAGE_SZ,
                   ngf=DEF_NGF,
                   ngpu=DEF_NGPU,
                   n_extra_layers=DEF_EXTRA_LAYERS,
                   force_reload=False):
    global generator

    if generator is None or force_reload:
        model_path = config_mgr.get_absolute_path("gan/" + model_file)

        generator = dcgan.DCGAN_G(image_sz, nz, NUM_TILES, ngf, ngpu, n_extra_layers)
        generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))


def apply_generator(latent_vector,
                    nz=DEF_LV_LEN): 
    with torch.no_grad():

      if type(latent_vector) is list:
          latent_vector = np.array(latent_vector)

      lv_tensor = torch.FloatTensor(latent_vector).view(1, nz, 1, 1)
      level_tensor = generator(Variable(lv_tensor))

      level_arr = level_tensor.data.cpu().numpy()
      level_arr = level_arr[:, :, :14, :28]  # Cut of rest to fit the 14x28 tile dimensions
      level_arr = np.argmax(level_arr, axis=1)

      # latent vector was processed with batch size of one so level_arr will be a length 1
      # array of levels
      return Level(data=level_arr[0])
