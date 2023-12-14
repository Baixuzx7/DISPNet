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
    if os.path.exists('./perstages/'):
        shutil.rmtree('./perstages/')

    """ Test """
    for idx in range(len(test_dataset)):
        print(idx,'/',len(test_dataset))
        lrpan, lrms, hrpan, hrms = test_dataset[idx]
        lrpan = lrpan.unsqueeze(0)
        lrms = lrms.unsqueeze(0)
        hrpan = hrpan.unsqueeze(0)
        hrms = hrms.unsqueeze(0)
        with torch.no_grad():            
            _,outs_list_lr,_,_,_,_ = PS_trainer.model(lrms.to(PS_trainer.device),lrpan.to(PS_trainer.device))
            _,outs_list_hr,_,_,_,_ = PS_trainer.model(hrms.to(PS_trainer.device),hrpan.to(PS_trainer.device))
            for i in range(len(outs_list_lr)):                
                if not os.path.exists('./perstages/stage_{}/'.format(i)):
                    os.makedirs('./perstages/stage_{}/lr/'.format(i))
                    os.makedirs('./perstages/stage_{}/hr/'.format(i))
                    

                image_lr_save = PS_trainer.tf2img(outs_list_lr[i]) * 255
                image_hr_save = PS_trainer.tf2img(outs_list_hr[i]) * 255

                PS_trainer.save_image(image=image_lr_save,save_path='./perstages/stage_{}/lr/'.format(i),name=idx)
                PS_trainer.save_image(image=image_hr_save,save_path='./perstages/stage_{}/hr/'.format(i),name=idx)

if __name__ == '__main__':
    """   Setting Parameters   """
    configs = BaseOptions()
    configs.print_options()
    configs.initialize()
    """   Train  """
    # train(configs)
    """   Test  """
    test(configs)
