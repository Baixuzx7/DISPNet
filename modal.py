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
                    os.makedirs('./priori/stage_{}/lrpan/L_pan/'.format(i))
                    os.makedirs('./priori/stage_{}/lrpan/R_pan/'.format(i))
                    os.makedirs('./priori/stage_{}/hrms/L_ms/'.format(i))
                    os.makedirs('./priori/stage_{}/hrms/R_ms/'.format(i))
                    pass

                input_ms_max = torch.max(hrms, dim=1, keepdim=True)[0]
                input_ms_img = torch.cat((input_ms_max, hrms), dim=1)
                ms_outs = PS_trainer.model.conv_Tp[i](input_ms_img)
                input_pan_img = lrpan.repeat(1,input_ms_img.size(1),1,1)
                pan_outs = PS_trainer.model.conv_Tp[i](input_pan_img)
                R_ms,L_ms = torch.sigmoid(ms_outs[:, 0:4, :, :]),torch.sigmoid(ms_outs[:, 4:5, :, :])   
                R_pan,L_pan = torch.sigmoid(pan_outs[:, 0:4, :, :]),torch.sigmoid(pan_outs[:, 4:5, :, :])  

                image_L_pan_save = PS_trainer.normalize_np(PS_trainer.tf2img(L_pan) * 255)
                image_R_pan_save = PS_trainer.normalize_np((PS_trainer.tf2img(R_pan) * 255)[:,:,-1])
                image_L_ms_save = PS_trainer.normalize_np(PS_trainer.tf2img(L_ms) * 255)
                image_R_ms_save = PS_trainer.normalize_np((PS_trainer.tf2img(R_ms) * 255)[:,:,[2,1,0]])

                """ ori  and rs"""

                PS_trainer.save_image(image=image_L_pan_save,save_path='./priori/stage_{}/lrpan/L_pan/'.format(i),name=idx)
                PS_trainer.save_image(image=image_R_pan_save,save_path='./priori/stage_{}/lrpan/R_pan/'.format(i),name=idx)
                PS_trainer.save_image(image=image_L_ms_save,save_path='./priori/stage_{}/hrms/L_ms/'.format(i),name=idx)
                PS_trainer.save_image(image=image_R_ms_save,save_path='./priori/stage_{}/hrms/R_ms/'.format(i),name=idx)



if __name__ == '__main__':
    """   Setting Parameters   """
    configs = BaseOptions()
    configs.print_options()
    configs.initialize()
    if os.path.exists('./priori/'):
        shutil.rmtree('./priori/')
    """   Train  """
    # train(configs)
    """   Test  """
    test_modal(configs)
