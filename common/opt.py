import argparse
from email.policy import default
import os
import math
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):

        self.parser.add_argument('--attempt', default=0, type=int)
        self.parser.add_argument('--model', default="HTNet_f_posev3", type=str)
        #pretrain
        self.parser.add_argument('--pretrain', action='store_true')
        self.parser.add_argument('--freeze', action='store_true')
        self.parser.add_argument('--cross', default=True, type=bool)
        self.parser.add_argument('--feature_dim', default=8, type=int)
        self.parser.add_argument('--queue_size', default=20480, type=int)
        self.parser.add_argument('-m','--momentum', default=0.999, type=float)
        self.parser.add_argument('--Temperature', default=0.5, type=float)  #0.07
        self.parser.add_argument('--lr_pre', default=0.001, type=float)
        self.parser.add_argument('--only0', action='store_true',help='only camera 0 ')
        self.parser.add_argument('--w', default=0.1, type=float)
        self.parser.add_argument('--bone_weight', default=0.2, type=float)
        self.parser.add_argument('--linear', action='store_true')
        self.parser.add_argument('--slice', default=1, type=int)
        self.parser.add_argument('-o','--optimizer', type=str, default='Adam', help='Name of your model')
        
        #model args
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--channel', default=240, type=int,help="Must be a multiple of 24")
        self.parser.add_argument('--frames','-f', type=int, default=243)
        self.parser.add_argument('--pad', type=int, default=121)
        self.parser.add_argument('-n','--model_name', type=str, default='your_model', help='Name of your model')
        self.parser.add_argument('--d_hid', default=256, type=int)
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)    
        self.parser.add_argument('--warmup_epochs', type=int, default=20)
        self.parser.add_argument('--lr1', default=1.75, type=float)
        self.parser.add_argument('--lr2', default=3.0, type=float)  
         
        self.parser.add_argument('--keep_frames', type=int, default=9, metavar='N', help='how many coefficients are kept')
        self.parser.add_argument('-frame-kept', '--number-of-kept-frames', default='27', type=int, metavar='N',
                            help='how many frames are kept')
        self.parser.add_argument('-coeff-kept', '--number-of-kept-coeffs', type=int, metavar='N', help='how many coefficients are kept')
        self.parser.add_argument('--hid', type=int, default=8, metavar='N', help='how many coefficients are kept')
            
            
        
        #train args
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=50)
        self.parser.add_argument('--batch_size','-b', type=int, default=256)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('--lr', type=float, default=0.0005)
        self.parser.add_argument('-lre','--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('-lrd', '--lr_decay', default=1.0, type=float)        
        self.parser.add_argument('-lrl','--lr_decay_large', type=float, default=0.5)     
        self.parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')   
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--drop',default=0.2, type=float)
        self.parser.add_argument('--seed',default=1, type=int)        
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)        
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--root_path', type=str, default='./dataset/',help='Put the dataset into this file')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)  
        self.parser.add_argument('--stride', default=1, type=int)       
        self.parser.add_argument('--lr_min',type=float,default=0,help='Min learn rate') 
        self.parser.add_argument('--subjects_train', type=str, default='S1,S5,S6,S7,S8') 
        self.parser.add_argument('--subjects_test', type=str, default='S9,S11') 
        
        
        # test args
        self.parser.add_argument('--test', type=int, default=1)
        self.parser.add_argument('--reload', action='store_true')
        self.parser.add_argument('-p','--previous_dir', type=str, default='./ckpt/your_model')
        self.parser.add_argument('--previous',type=str,default='ckpt')
        self.parser.add_argument('--previous_best', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('--viz', type=str, default='try')
        
        #refine
        self.parser.add_argument('--refine', action='store_true')
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--lr_refine', type=float, default=1e-5)
        self.parser.add_argument('--refine_train_reload', action='store_true')
        self.parser.add_argument('-tds', '--t_downsample', type=int, default=3)
        self.parser.add_argument('--refine_test_reload', action='store_true')
        self.parser.add_argument('-previous_refine_name', type=str, default='')


    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        self.opt.pad = (self.opt.frames-1) // 2
        # self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        # self.opt.subjects_test = 'S9,S11'

        if self.opt.train:
            self.opt.checkpoint = 'ckpt/' + self.opt.model_name
            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)


            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')

        return self.opt






