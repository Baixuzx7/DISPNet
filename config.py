import argparse
import torchvision
import os

class BaseOptions():
    """This class defines options used during both training and test time"""

    def __init__(self):
        parser = argparse.ArgumentParser()
        """Define the common options that are used in both training and test."""
        # Device parameters
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--sensor', type=str, default='QuickBird')
        # Data parameters
        parser.add_argument('--train_root',type=str, default='./data/TrainFolder')
        parser.add_argument('--valid_root',type=str, default='./data/ValidFolder')
        parser.add_argument('--test_root',type=str, default='./data/TestFolder')

        """ Pansharpening Network """
        # train parameters
        parser.add_argument('--n_epochs', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--save_per_epoch', type=int, default=1)
        parser.add_argument('--test_per_epoch', type=int, default=499)
        parser.add_argument('--isshuffle', type=bool, default=True)
        parser.add_argument('--iscontinue', type=bool, default=False)
        parser.add_argument('--continue_load_name_PS', type=str, default='pretrain')
        parser.add_argument('--continue_load_name_ASE', type=str, default='pretrain')
        # Optimizer parameters
        parser.add_argument('--lr_PS', type=float, default=1e-3)
        parser.add_argument('--beta1_PS', type=float, default=0.9)
        parser.add_argument('--beta2_PS', type=float, default=0.999)
        parser.add_argument('--initial_loss_PS', type=float, default=999)
        # Scheduler parameters
        parser.add_argument('--lr_scheduler_step_PS', type=int, default=200)
        parser.add_argument('--lr_scheduler_decay_PS', type=float, default=0.3)

        """ Adaptive Stage Exiting Regressor """
        # Optimizer parameters
        parser.add_argument('--lr_ASE', type=float, default=1e-3)
        parser.add_argument('--beta1_ASE', type=float, default=0.9)
        parser.add_argument('--beta2_ASE', type=float, default=0.999)
        parser.add_argument('--initial_loss_ASE', type=float, default=999)
        # Scheduler parameters
        parser.add_argument('--lr_scheduler_step_ASE', type=int, default=100)
        parser.add_argument('--lr_scheduler_decay_ASE', type=float, default=0.5)

        # Save directory parameters
        parser.add_argument('--net_params_dir_PS', type=str, default='./checkpoint/PS/net')
        parser.add_argument('--opt_params_dir_PS', type=str, default='./checkpoint/PS/opt')
        parser.add_argument('--net_params_dir_ASE', type=str, default='./checkpoint/ASE/net')
        parser.add_argument('--opt_params_dir_ASE', type=str, default='./checkpoint/ASE/opt')
        
        parser.add_argument('--result_ori_hr', type=str, default='./result/ori/hr')
        parser.add_argument('--result_ori_lr', type=str, default='./result/ori/lr')
        parser.add_argument('--result_rs_hr', type=str, default='./result/rs/hr')
        parser.add_argument('--result_rs_lr', type=str, default='./result/rs/lr')


        parser.add_argument('--writer_dir_PS', type=str, default='./blog/PS')
        parser.add_argument('--writer_dir_ASE', type=str, default='./blog/ASE')
        
        # test parameters
        parser.add_argument('--test_epoch_params', type=int, default=299)
        self.opt = parser.parse_args()
        self.transform = {
            'train': torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor()
                ]
            ),
            'test': torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor()
                ]
            )}

    def initialize(self):
        if not os.path.exists(self.opt.net_params_dir_PS):
            os.makedirs(self.opt.net_params_dir_PS)
        if not os.path.exists(self.opt.opt_params_dir_PS):
            os.makedirs(self.opt.opt_params_dir_PS)
        if not os.path.exists(self.opt.writer_dir_PS):
            os.makedirs(self.opt.writer_dir_PS)
        if not os.path.exists(self.opt.net_params_dir_ASE):
            os.makedirs(self.opt.net_params_dir_ASE)
        if not os.path.exists(self.opt.opt_params_dir_ASE):
            os.makedirs(self.opt.opt_params_dir_ASE)
        if not os.path.exists(self.opt.writer_dir_ASE):
            os.makedirs(self.opt.writer_dir_ASE)
        if not os.path.exists(self.opt.result_ori_hr):
            os.makedirs(self.opt.result_ori_hr)
        if not os.path.exists(self.opt.result_ori_lr):
            os.makedirs(self.opt.result_ori_lr)
        if not os.path.exists(self.opt.result_rs_hr):
            os.makedirs(self.opt.result_rs_hr)
        if not os.path.exists(self.opt.result_rs_lr):
            os.makedirs(self.opt.result_rs_lr)

    def print_options(self):
        """Print and save options"""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)


if __name__ == "__main__":
    Baseopt = BaseOptions()
    Baseopt.print_options()