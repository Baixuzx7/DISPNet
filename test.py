import torch
import numpy as np
import random
from tqdm import tqdm
from config import BaseOptions
from model import PSNetwork,PSTrainer
from dataset import PanSharpeningDataset
from eval import TensorToIMage,ERGAS,RASE,RMSE,QAVE
import os
import shutil


def test(configs):
    test_dataset = PanSharpeningDataset(configs.opt.test_root, transform=configs.transform['test'])
    PS_trainer = PSTrainer(model = PSNetwork(), opt = configs.opt)
    PS_trainer.load_model_parameters(configs.opt.net_params_dir_PS,'Best')
    """ Test """
    for idx in range(len(test_dataset)):
        print(idx,'/',len(test_dataset))
        lrpan, lrms, hrpan, hrms = test_dataset[idx]
        lrpan = lrpan.unsqueeze(0)
        lrms = lrms.unsqueeze(0)
        hrpan = hrpan.unsqueeze(0)
        hrms = hrms.unsqueeze(0)
        with torch.no_grad():            
            image_lr,_,_,_,_,_ = PS_trainer.model(lrms.to(PS_trainer.device),lrpan.to(PS_trainer.device))
            image_hr,_,_,_,_,_ = PS_trainer.model(hrms.to(PS_trainer.device),hrpan.to(PS_trainer.device))
            image_lr_save = PS_trainer.tf2img(image_lr) * 255
            image_hr_save = PS_trainer.tf2img(image_hr) * 255
            """ ori  and rs"""
            PS_trainer.save_image(image=image_lr_save,save_path=PS_trainer.opt.result_ori_lr,name=idx)
            PS_trainer.save_image(image=image_hr_save,save_path=PS_trainer.opt.result_ori_hr,name=idx)

if __name__ == '__main__':
    """   Setting Parameters   """
    configs = BaseOptions()
    configs.print_options()
    configs.initialize()
    """   Train  """
    # train(configs)
    """   Test  """
    test(configs)
