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

def train(configs):
    train_dataset = PanSharpeningDataset(configs.opt.train_root, transform=configs.transform['train'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.opt.batch_size, shuffle=configs.opt.isshuffle)
    valid_dataset = PanSharpeningDataset(configs.opt.valid_root, transform=configs.transform['train'])
    PS_trainer = PSTrainer(model = PSNetwork(), opt = configs.opt)
    PS_Loss_Max = configs.opt.initial_loss_PS
    for epoch in range(configs.opt.n_epochs):
        """ Train """
        PS_trainer.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for (image_pan, image_ms, image_pan_label, image_ms_label) in tepoch:
                """ PStrainer """
                PS_trainer.optimizer.zero_grad()
                PSloss = PS_trainer(image_ms,image_pan,image_ms_label)
                PSloss.backward()
                PS_trainer.optimizer.step()
                tepoch.set_description_str('Epoch:{}/{}  PSLoss: {:.6f} '.format(epoch, configs.opt.n_epochs,PSloss.item()))
                pass
        PS_trainer.scheduler.step()
        """ Valid """ 
        PS_trainer.eval()
        if epoch % configs.opt.save_per_epoch == 0:
            er,rm,ra,qa = [],[],[],[]
            for j in range(len(valid_dataset)):
                image_pan, image_ms, image_pan_label, image_ms_label = valid_dataset[j]
                with torch.no_grad():
                    lr_tf,_,_,_,_,_ = PS_trainer.model(image_ms.unsqueeze(0).to(PS_trainer.device),image_pan.unsqueeze(0).to(PS_trainer.device))
                    lr = TensorToIMage(lr_tf.squeeze(0).cpu())
                    ms = TensorToIMage(image_ms_label)
                    er.append(ERGAS(ms,lr))
                    rm.append(RMSE(ms,lr))
                    ra.append(RASE(ms,lr))
                    qa.append(QAVE(ms,lr))
                    pass
                pass
            er_mean = np.asarray(er).mean()
            rm_mean = np.asarray(rm).mean()
            ra_mean = np.asarray(ra).mean()
            qa_mean = np.asarray(qa).mean()
            PS_Loss_Eval = rm_mean
            PS_trainer.writer.add_scalar('PS_ERGAS',er_mean,global_step=epoch)
            PS_trainer.writer.add_scalar('PS_RMSE' ,rm_mean,global_step=epoch)
            PS_trainer.writer.add_scalar('PS_RASE' ,ra_mean,global_step=epoch)
            PS_trainer.writer.add_scalar('PS_QAVE' ,qa_mean,global_step=epoch)
            print('Evaluate ---> PSERGAS :  {:.4f} PSRMSE :  {:.4f} PSRASE :  {:.6f} PSQAVE :  {:.6f}  '.format(er_mean,rm_mean,ra_mean,qa_mean))
            """ PStrainer SAVE Parameters """
            PS_trainer.save_model_parameters(configs.opt.net_params_dir_PS,epoch)
            PS_trainer.save_optim_parameters(configs.opt.opt_params_dir_PS,epoch)
            """ SAVE BEST Parameters"""
            if PS_Loss_Eval < PS_Loss_Max:
                PS_Loss_Max = PS_Loss_Eval
                PS_trainer.save_model_parameters(configs.opt.net_params_dir_PS,'Best')
                PS_trainer.save_optim_parameters(configs.opt.opt_params_dir_PS,'Best')
                pass
        """ Test """
        if epoch > 1 and epoch % configs.opt.test_per_epoch == 0:
            # remeber to recify the configs.opt.test_per_epoch
            test(configs)


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


""" Seed Setting"""
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    """   Setting Parameters   """
    setup_seed(45)
    torch.cuda.empty_cache()
    configs = BaseOptions()
    configs.print_options()
    configs.initialize()
    """   Romve the abundant files """
    if configs.opt.iscontinue is False:
        if os.path.exists('./checkpoint'):
            shutil.rmtree('./checkpoint')
        if os.path.exists('./result'):
            shutil.rmtree('./result')
        if os.path.exists('./blog'):
            shutil.rmtree('./blog')
        pass
    """   Train  """
    train(configs)
    """   Test  """
    # test(configs)
