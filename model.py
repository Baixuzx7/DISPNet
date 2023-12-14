from modules import *


class PSNetwork(nn.Module):
    def __init__(self, mid_channels=64, T = 4):
        super().__init__()

        print("Now: PSNetwork_V4")
        self.n_stage = T
        self.up_factor = 4
        kSize = 3
        # U
        self.conv_u = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(4 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        self.conv_Th = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(4 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        self.conv_Tt = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(4 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        self.conv_Tu = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(4 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        self.conv_Tp = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(5 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 5, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        # V
        self.conv_v = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(4 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        self.conv_down_v = nn.ModuleList([nn.Sequential(*[Conv_down(4, mid_channels, self.up_factor)]) for _ in range(T)])
        self.conv_up_v = nn.ModuleList([nn.Sequential(*[Conv_up(4, mid_channels, self.up_factor)]) for _ in range(T)])
        self.conv_down_p = nn.ModuleList([nn.Sequential(*[                         
            Conv_down(1, mid_channels, self.up_factor),
            nn.Conv2d(1 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        
        self.conv_Tv = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(4 ,64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)]) for _ in range(T)])
        # X
        self.conv_up = Conv_up(4, mid_channels, self.up_factor)
        self.conv_down = Conv_down(4, mid_channels, self.up_factor)

        # Parameters 
        self.delta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.zeta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(T)])

    def forward(self, lms, pan):
        hms = torch.nn.functional.interpolate(lms, scale_factor=self.up_factor, mode='bilinear', align_corners=False)   # B 4 256 256
        x = hms
        uk_list,vk_list,outs_list = [],[],[]
        R_ms_stack,L_ms_stack,R_pan_stack,L_pan_stack = [],[],[],[]
        for i in range(self.n_stage):
            """ U sub problem"""
            # priori variable
            uk = self.conv_u[i](x)         # B 4 h w
            Thuk = self.conv_Th[i](uk)     # B 4 h w x
            # intrinsic variable
            input_ms_max = torch.max(x, dim=1, keepdim=True)[0]
            input_ms_img = torch.cat((input_ms_max, x), dim=1)
            ms_outs = self.conv_Tp[i](input_ms_img)
            input_pan_img = pan.repeat(1,input_ms_img.size(1),1,1)
            pan_outs = self.conv_Tp[i](input_pan_img)
            R_ms,L_ms = torch.sigmoid(ms_outs[:, 0:4, :, :]),torch.sigmoid(ms_outs[:, 4:5, :, :])   
            R_pan,L_pan = torch.sigmoid(pan_outs[:, 0:4, :, :]),torch.sigmoid(pan_outs[:, 4:5, :, :])  
        
            decode_u = self.conv_Tt[i](Thuk - L_pan) + uk - x # B 4 h w
            decode_u = self.conv_Tu[i](decode_u) + uk          # B 4 h w
            uk_list.append(decode_u)
            """ V sub problem"""
            vk = self.conv_v[i](x)         # B 4 h w
            DBvk = self.conv_down_v[i](vk)
            decode_v = self.conv_up_v[i](DBvk - self.conv_down_p[i](pan)) + vk - x  # B 4 h w
            decode_v = self.conv_Tv[i](decode_v) + vk  # B 4 h w
            vk_list.append(decode_v)
            """ X sub problem"""
            x = x - self.delta[i]*(self.conv_up(self.conv_down(x)-lms)+self.eta[i]*(x-decode_u))-self.zeta[i]*(x - decode_v)
            outs_list.append(x)
            # iteration
            R_ms_stack.append(R_ms)
            L_ms_stack.append(L_ms)
            R_pan_stack.append(R_pan)
            L_pan_stack.append(L_pan)

        return x,outs_list,R_ms_stack,L_ms_stack,R_pan_stack,L_pan_stack
        # , uk_list[-1], vk_list[-1] # , decoder_list, fea_list
    

    def initialize(self,conv_type = 'trunc',bias_type = None ,bn_type = None):
        """ Initialize the parameters"""        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if conv_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                elif conv_type == 'trunc':
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.05)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                else:
                    nn.init.constant_(m.weight, val=0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            pass


""" Pansharpening Trainer """
class PSTrainer(nn.Module):
    def __init__(self,model,opt):
        super(PSTrainer,self).__init__()
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=opt.lr_PS,betas=(opt.beta1_PS,opt.beta2_PS))        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_scheduler_step_PS, gamma=opt.lr_scheduler_decay_PS)
        self.writer = SummaryWriter(opt.writer_dir_PS)
        self.criterion = nn.MSELoss().to(self.device) # SSIM_Loss() # nn.MSELoss().to(self.device)
        if opt.iscontinue:
            print('Continue Training --> PStrainer Loads Parameters : {}.pth'.format(opt.continue_load_name_PS))
            self.load_model_parameters(opt.net_params_dir_PS,name = opt.continue_load_name_PS)
            self.load_optim_parameters(opt.opt_params_dir_PS,name = opt.continue_load_name_PS)
        else:
            print('Initializing --> PStrainer Gets Parameters')
            # self.model.initialize()
 
    def forward(self,ms,pan,gt):
        x,H_stack,R_ms_stack,L_ms_stack,R_pan_stack,L_pan_stack = self.model(ms.to(self.device),pan.to(self.device))
        # loss = self.criterion(H.to(self.device),gt.to(self.device))
        loss_sharpen = self.sharpen_loss_fn(H_stack,gt)
        loss_intrinsic = self.intrinsic_loss_fn(R_ms_stack,L_ms_stack,R_pan_stack,L_pan_stack,gt,pan)
        loss = loss_sharpen + 0.1 * loss_intrinsic
        return loss

    def sharpen_loss_fn(self,H_stack,gt):
        loss_sharpen = 0
        for i in range(len(H_stack)):
            loss_sharpen = loss_sharpen + self.criterion(H_stack[i].to(self.device),gt.to(self.device)) 
        return loss_sharpen/len(H_stack)

    def intrinsic_loss_fn(self,R_ms_stack,L_ms_stack,R_pan_stack,L_pan_stack,gt,pan):
        """
            # ms  : multi-spectral image n c h w
            # pan : panchromatic image   n c h w
        """
        loss_intrinsic = 0
        for i in range(len(L_pan_stack)):
            R_ms,I_ms,R_pan,I_pan = R_ms_stack[i].to(self.device),L_ms_stack[i].to(self.device),R_pan_stack[i].to(self.device),L_pan_stack[i].to(self.device)
            ms = gt.to(self.device)
            tpan = pan.repeat(1,ms.size(1),1,1).to(self.device)
            I_ms_cat  = I_ms.repeat(1,R_ms.size(1),1,1)
            I_pan_cat = I_pan.repeat(1,R_pan.size(1),1,1)
            self.recon_loss_low  = F.l1_loss(R_ms * I_ms_cat, ms)
            self.recon_loss_high = F.l1_loss(R_pan * I_pan_cat, tpan)
            self.recon_loss_mutal_low  = F.l1_loss(R_pan * I_ms_cat, ms)
            self.recon_loss_mutal_high = F.l1_loss(R_ms * I_pan_cat, tpan)
            self.equal_I_loss = F.l1_loss(I_ms,I_pan.detach())
            self.Rsmooth_loss_low   = self.smooth(I_ms, R_ms)
            self.Rsmooth_loss_high  = self.smooth(I_pan, R_pan)
            loss_intrinsic = loss_intrinsic + self.recon_loss_low + self.recon_loss_high + \
                            0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + \
                            0.1 * self.Rsmooth_loss_low + 0.1 * self.Rsmooth_loss_high + 0.01 * self.equal_I_loss

        return loss_intrinsic/len(L_pan_stack)
    

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        """ input_I is high frequency """
        """ input_R is  low frequency """
        input_R = torch.mean(input_R,dim=1).unsqueeze(1)
        return torch.mean(self.gradient(input_R, "x") * torch.exp(-10 * self.ave_gradient(input_I, "x")) +
                          self.gradient(input_R, "y") * torch.exp(-10 * self.ave_gradient(input_I, "y")))

    
    def save_model_parameters(self,save_path,name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(),os.path.join(save_path,'{}.pth'.format(name)))

    def load_model_parameters(self,load_path,name):
        if not os.path.exists(load_path):
            exit('No such file, Please Check the path : ', load_path,name,'.pth')
        else:
            print('Loading Model',os.path.join(load_path,'{}.pth'.format(name)))
        self.model.load_state_dict(torch.load(os.path.join(load_path,'{}.pth'.format(name)),self.device))

    def save_optim_parameters(self,save_path,name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.optimizer.state_dict(),os.path.join(save_path,'{}.pth'.format(name)))

    def load_optim_parameters(self,load_path,name):
        if not os.path.exists(load_path):
            exit('No such file, Please Check the path : ', load_path,name,'.pth')
        else:
            print('Loading Optim',os.path.join(load_path,'{}.pth'.format(name)))
        self.optimizer.load_state_dict(torch.load(os.path.join(load_path,'{}.pth'.format(name)),self.device))

    def img2tf(self,image_np):
        if len(image_np.shape) == 2:
            image_tf = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)
        else:
            image_tf = torch.from_numpy(image_np).permute(2,0,1).unsqueeze(0)
        
        return image_tf

    def tf2img(self,image_tf):
        n,c,h,w = image_tf.size()
        assert n == 1
        if c == 1:
            image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            image_np = image_tf.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        
        return image_np
    
    def normalize_np(self,image_np):
        image_np_db = image_np.astype(np.float32)
        if len(image_np.shape) > 2:
            image_norm = np.zeros_like(image_np_db)
            for i in range(image_np_db.shape[-1]):
                maximum = np.max(image_np_db[:,:,i])
                minimum = np.min(image_np_db[:,:,i])
                image_norm[:,:,i] = (image_np_db[:,:,i] - minimum) / (maximum - minimum + 1e-23) * 255
        else:
            image_norm = np.zeros_like(image_np_db)
            maximum = np.max(image_np_db)
            minimum = np.min(image_np_db)
            image_norm = (image_np_db - minimum) / (maximum - minimum + 1e-23) * 255
        return image_norm.astype(np.uint8)

    def save_image(self,image,save_path,name,type = 'tif'):
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))
        imageio.imwrite(os.path.join(save_path,'{}.{}'.format(name,type)),image.astype(np.uint8))


""" Main Procedure """

if __name__ == '__main__':
    print('Hello World')
    ms = torch.rand([1,4,32,32])
    pan = torch.rand([1,1,128,128])
    gt = torch.rand([1,4,128,128])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PSNetwork().to(device)
    net.initialize()
    x,H,_,_,_,_ = net(ms.to(device),pan.to(device))
    sharpen_result = x
    print('sharpen_result.size()',sharpen_result.size())
    print('sharpen_result.max()',sharpen_result.max())
    print('sharpen_result.min()',sharpen_result.min())
    
    # configs = BaseOptions()
    # configs.print_options()
    # configs.initialize()
    # pstrainer = PSTrainer(model = net,opt = configs.opt)
    # print('Finished')
    # loss = pstrainer(ms,pan,gt)
    # print(loss.item())
    # loss.backward()

