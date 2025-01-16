import sys
import torch
sys.path.insert(0, '..')

class BaseConfig():
    def __init__(self):

        #---------------------
        # Training Parameters
        #---------------------
        self.data_root = '/home/jiao301/user1/yejixun/RSIE-main/zhanglab'
        self.print_freq = 10
        self.device = 'cuda:0'
        self.epochs = 200
        self.lr = 1e-4#1e-3 # learning rate
        self.batch_size = 16
        self.test_batch_size = 2
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR
        self.scheduler_args = dict(milestones=[200, 300], gamma=0.2)

        # GAN
        self.discriminator_type = 'basic'
        self.enbale_gan = 0 #100
        self.lambda_gp = 10
        self.size = 4
        self.n_critic = 1
        self.sample_interval = 1000
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR
        self.scheduler_args_d = dict(milestones=[200-self.enbale_gan, 300-self.enbale_gan], gamma=0.2)

        # model
        self.num_patch = 2 #4
        self.level = 4 #
        self.shrink_thres = 0.0005
        self.initial_combine = 2
        self.drop = 0.
        self.dist = True
        self.num_slots = 1000
        self.mem_num_slots = 500
        self.memory_channel = 2048
        self.img_size = 128
        self.mask_ratio = 0.95
        self.ops = ['concat', 'concat', 'none', 'none']
        self.decoder_memory = [None,
                               None,
                               dict(type='MemoryMatrixBlock',dim=128,size=16,multiplier=64, num_memory=self.num_patch**2),
                               dict(type='MemoryMatrixBlock',dim=256,size=8,multiplier=16, num_memory=self.num_patch**2)]
        # self.decoder_memory = [  #dict(type='MemoryMatrixBlock', dim=16, size=128, multiplier=64*16*4,num_memory=self.num_patch ** 2),
        #                         None,
        #                        dict(type='MemoryMatrixBlock', dim=32, size=64, multiplier=64*16,num_memory=self.num_patch ** 2),
        #                        # None,
        #                        dict(type='MemoryMatrixBlock', dim=64, size=32, multiplier=64,num_memory=self.num_patch ** 2),
        #                        # None,
        #                        dict(type='MemoryMatrixBlock', dim=256, size=8, multiplier=16,num_memory=self.num_patch ** 2)]

        # loss weight
        self.t_w = 0.5
        self.recon_w = 1.
        self.dist_w = 0.1
        self.g_w = 0.0005
        self.d_w = 1.

        # misc
        self.disable_tqdm = False
        self.dataset_name = 'zhang'
        self.early_stop = 200
        self.limit = None

        # alert
        self.alert = None#Alert(lambda1=1., lambda2=1.)