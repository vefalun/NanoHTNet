# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import math
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    #pretrain
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--cross', default=True, type=bool)
    parser.add_argument('--feature_dim', default=3, type=int)
    parser.add_argument('--queue_size', default=15360, type=int)
    parser.add_argument('-m','--momentum', default=0.999, type=float)
    parser.add_argument('--Temperature', default=0.07, type=float)  #0.07
    parser.add_argument('--lr_pre', default=0.001, type=float)
    parser.add_argument('--only0', action='store_true',help='only camera 0 ')
    parser.add_argument('--w', default=0.1, type=float)
    parser.add_argument('--bone_weight', default=0.2, type=float)
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('-o','--optimizer', type=str, default='Adam', help='Name of your model')


    # General arguments
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--channel', default=240, type=int,help="Must be a multiple of 24")
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('-n','--model_name', type=str, default='your_model', help='Name of your model')
    parser.add_argument('--d_hid', default=1024, type=int)
    parser.add_argument('--n_joints', type=int, default=17)
    parser.add_argument('--out_joints', type=int, default=17)
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--out_channels', type=int, default=3)        
    parser.add_argument('--root_joint', type=int, default=0)
  
    
    
    #train args
    parser.add_argument('--gpu', default='0', type=str, help='')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--nepoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='h36m')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--large_decay_epoch', type=int, default=5)
    parser.add_argument('-lrd', '--lr_decay', default=0.99, type=float)        
    parser.add_argument('--lr_decay_large', type=float, default=0.5)  
    parser.add_argument('--lr_decay_epoch', type=int, default=5)    
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')   
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out_all', type=int, default=1)
    parser.add_argument('--drop',default=0.2, type=float)
    parser.add_argument('--seed',default=1, type=int)        
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--test_augmentation', type=bool, default=True)        
    parser.add_argument('--reverse_augmentation', type=bool, default=False)
    parser.add_argument('--root_path', type=str, default='./dataset/',help='Put the dataset into this file')
    parser.add_argument('-a', '--actions', default='*', type=str)
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--subset', default=1, type=float)  
    parser.add_argument('--stride', default=1, type=float)       
    parser.add_argument('--lr_min',type=float,default=0,help='Min learn rate') 
    
    
    # test args
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('-p','--previous_dir', type=str, default='./ckpt/your_model')
    parser.add_argument('--previous',type=str,default='ckpt')
    parser.add_argument('--previous_best', type=float, default= math.inf)
    parser.add_argument('-previous_name', type=str, default='')
    parser.add_argument('--viz', type=str, default='try')
    
    #refine
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--crop_uv', type=int, default=0)
    parser.add_argument('--lr_refine', type=float, default=1e-5)
    parser.add_argument('--refine_train_reload', action='store_true')
    parser.add_argument('--refine_test_reload', action='store_true')
    parser.add_argument('-previous_refine_name', type=str, default='')
    
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-c', '--checkpoint', default='checkpoint/', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-l', '--log', default='checkpoint/', type=str, metavar='PATH',
                        help='log file directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--export-training-curves', default = 1, type=int, help='save training curves as .png images')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    
    # # Visualization
    # parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    # parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    # parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    # parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    # parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    # parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    # parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    # parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    # parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    # parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    # parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    # parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')
    # parser.add_argument('--compare', action='store_true', default=False, help='Whether to compare with other methods e.g. Poseformer')
    # # parser.add_argument('-comchk', type=str, default='/mnt/data3/home/zjl/workspace/3dpose/PoseFormer/checkpoint/detected81f.bin', help='checkpoint of comparison methods')


    
    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()
        
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()
    
    if args.train:
        args.checkpoint = 'ckpt/' + args.model_name
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

        opt = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))

        file_name = os.path.join(args.checkpoint, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(opt.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

    return args