import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from datasets import fetch_dataloader
from datasets import fetch_videoloader
import random
from utils.log import Logger

from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
import sys
import json
import cv2
import custom_data


from utils.visualize_final import visvid as visualize

import matplotlib.cm as cm


warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt
import pdb

class DraftRefine():

    def __init__(self, args):

        self.config = args
        torch.backends.cudnn.benchmark = True
        torch.multiprocessing.set_sharing_strategy('file_system')
        self._build()


    def eval(self):
       


        
        
        config = self.config

        if not os.path.exists(config.imwrite_dir):
            os.mkdir(config.imwrite_dir)
            

        with torch.no_grad():

            model = self.model.eval()
            epoch_tested = self.config.testing.ckpt_epoch

            if epoch_tested == 0 or epoch_tested == '0':
                checkpoint = torch.load(self.config.corr_weights)
                dict = {k.replace('module.', ''): checkpoint['model'][k] for k in checkpoint['model']}
                model.module.corr.load_state_dict(dict)
            else:
                ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
                print("Evaluation...")
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['model'])


            if not os.path.exists(os.path.join(self.evaldir, 'epoch' + str(epoch_tested))):
                os.mkdir(os.path.join(self.evaldir, 'epoch' + str(epoch_tested)))
            if not os.path.exists(os.path.join(self.evaldir, 'epoch' + str(epoch_tested), 'jsons')):
                os.mkdir(os.path.join(self.evaldir, 'epoch' + str(epoch_tested), 'jsons'))
           
                
          
            model.eval()
            s0 = custom_data.DataSample('/kaggle/working/AnimeInbet/data/ml100_norm/all/frames/chip_abe/Image0027.png', '/kaggle/working/AnimeInbet/data/ml100_norm/all/labels/chip_abe/Line0027.json')
            s1 = custom_data.DataSample('/kaggle/working/AnimeInbet/data/ml100_norm/all/frames/chip_abe/Image0032.png', '/kaggle/working/AnimeInbet/data/ml100_norm/all/labels/chip_abe/Line0032.json')
            data = custom_data.make_model_input(s0, s1)

            pred = model(data)
            images = visualize(pred)

            for ii, image in enumerate(images):
                cv2.imwrite(os.path.join(config.imwrite_dir, f'result_{ii}.png'), image)

             




            



    def _build(self):
        
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        self._build_optimizer()



    def _build_model(self):
        """ Define Model """
        config = self.config 
        if hasattr(config.model, 'name'):
            print(f'Experiment Using {config.model.name}')
            model_class = getattr(models, config.model.name)
            model = model_class(config.model)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
        self.model = model.cuda()



   
    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model.module.parameters(),
                                             ),
                                             **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

        

    def _dir_setting(self):
        self.expname = self.config.expname
        # self.experiment_dir = os.path.join("/mnt/cache/syli/inbetween", "experiments")

        self.experiment_dir = 'experiments'
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        self.viddir = os.path.join(self.expdir, "video")
        if not os.path.exists(self.viddir):
            os.mkdir(self.viddir)

        


        
