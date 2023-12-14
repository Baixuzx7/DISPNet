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

def test_modal(configs):
    test_dataset = PanSharpeningDataset(configs.opt.test_root, transform=configs.transform['test'])
    PS_trainer = PSTrainer(model = PSNetwork(), opt = configs.opt)
    PS_trainer.load_model_parameters(configs.opt.net_params_dir_PS,'Best')
    """ Test """
    for idx in range(len(test_dataset)): #len(test_dataset)
        print(idx,'/',len(test_dataset))
        lrpan, lrms, hrpan, hrms = test_dataset[idx]
        lrpan = lrpan.unsqueeze(0).to(PS_trainer.device)
        lrms = lrms.unsqueeze(0).to(PS_trainer.device)
        hrpan = hrpan.unsqueeze(0).to(PS_trainer.device)
        hrms = hrms.unsqueeze(0).to(PS_trainer.device)
        with torch.no_grad():            
            for i in range(4):
                if not os.path.exists('./priori/stage_{}/'.format(i)):
                    os.makedirs('./intrinsic/stage_{}/MS/'.format(i))
                    os.makedirs('./intrinsic/stage_{}/PAN/'.format(i))
                    pass

                input_ms_max = torch.max(hrms, dim=1, keepdim=True)[0]
                input_ms_img = torch.cat((input_ms_max, hrms), dim=1)
                ms_outs = PS_trainer.model.conv_Tp[i](input_ms_img)
                input_pan_img = lrpan.repeat(1,input_ms_img.size(1),1,1)
                pan_outs = PS_trainer.model.conv_Tp[i](input_pan_img)
                R_ms,L_ms = torch.sigmoid(ms_outs[:, 0:4, :, :]),torch.sigmoid(ms_outs[:, 4:5, :, :])   
                R_pan,L_pan = torch.sigmoid(pan_outs[:, 0:4, :, :]),torch.sigmoid(pan_outs[:, 4:5, :, :])

                ms_sys = R_ms * (L_ms.repeat(1,R_ms.size(1),1,1))
                pan_sys = R_pan * (L_pan.repeat(1,R_pan.size(1),1,1))

                image_ms_sys = PS_trainer.tf2img(ms_sys) * 255
                image_pan_sys = PS_trainer.tf2img(pan_sys) * 255

                PS_trainer.save_image(image=image_ms_sys,save_path='./intrinsic/stage_{}/MS/'.format(i),name=idx)
                PS_trainer.save_image(image=image_pan_sys,save_path='./intrinsic/stage_{}/PAN/'.format(i),name=idx)


if __name__ == '__main__':
    """   Setting Parameters   """
    configs = BaseOptions()
    configs.print_options()
    configs.initialize()
    if os.path.exists('./intrinsic/'):
        shutil.rmtree('./intrinsic/')
    """   Train  """
    # train(configs)
    """   Test  """
    test_modal(configs)
