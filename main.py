# import os
# import glob
# import torch
# import random
# import logging
# import numpy as np
# from tqdm import tqdm
# import torch.utils.data
# import torch.optim as optim
# import torch.nn as nn
# import math

# from common.opt import opts
# from common.utils import *
# from common.load_data_h36m import Fusion
# from common.h36m_dataset import Human36mDataset

# from model.GCN_conv import adj_mx_from_skeleton
# from model.poseclr_f import PoseCLR
# from model.trans import HTNet as NanoHTNet
# import common.loss as eval_loss
# from thop import profile
# from tensorboardX import SummaryWriter

# def train(opt, train_loader, model, optimizer):
#     return step('train', opt, train_loader, model, optimizer)

# def val(opt, val_loader, model):
#     with torch.no_grad():
#         return step('test', opt, val_loader, model)

# def step(split, opt, data_loader, model, optimizer=None):
#     if split == 'train':
#         model.train()
#     else:
#         model.eval()

#     loss_all = AccumLoss()
#     action_error_sum = define_error_list(define_actions(opt.actions))
#     mid_frame = opt.keep_frames // 2

#     for data in tqdm(data_loader, 0):
#         batch_cam, gt_3D, input_2D, action, subject, _, _, _ = data
#         input_2D, gt_3D, batch_cam = get_varialbe(split, [input_2D, gt_3D, batch_cam])

#         if split == 'train':
#             output_3D = model(input_2D)
#         else: 
#             input_2D, output_3D = input_augmentation(input_2D, model)
        
#         out_target = gt_3D.clone()
#         out_target[:, :, 0] = 0

#         if split == 'train':
#             if opt.model in ["HTNet_f_posev2", "HTNet_f_posev3"]:
#                 index = torch.arange((opt.frames - 1) // 2 - opt.keep_frames // 2, (opt.frames - 1) // 2 + opt.keep_frames // 2 + 1)
#                 out_target = out_target[:, index, :, :]
            
#             if opt.attempt == 10:
#                 w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
#                 loss_w_mpjpe = eval_loss.weighted_mpjpe(output_3D, out_target, w_mpjpe)
#                 loss_temporal = eval_loss.temporal_consistency(output_3D, out_target)
#                 loss_mean_velocity = eval_loss.mean_velocity(output_3D, out_target, axis=1)
#                 loss = loss_w_mpjpe + 0.5 * loss_temporal + 2.0 * loss_mean_velocity
#             else:
#                 loss = mpjpe_cal(output_3D, out_target)
            
#             N = input_2D.size(0)
#             loss_all.update(loss.detach().cpu().numpy() * N, N)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
        
#         else: # 'test' split
#             output_3D = output_3D[:, mid_frame if opt.model in ["HTNet_f_posev2", "HTNet_f_posev3"] else opt.pad].unsqueeze(1)
#             output_3D[:, :, 0, :] = 0
#             action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)

#     if split == 'train':
#         return loss_all.avg
#     elif split == 'test':
#         p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
#         return p1, p2
    
#     return None, None

# def input_augmentation(input_2D, model):
#     joints_left = [4, 5, 6, 11, 12, 13]
#     joints_right = [1, 2, 3, 14, 15, 16]
    
#     input_2D_non_flip = input_2D[:, 0]
#     input_2D_flip = input_2D[:, 1]
    
#     output_3D_non_flip = model(input_2D_non_flip)
#     output_3D_flip = model(input_2D_flip)
    
#     output_3D_flip[:, :, :, 0] *= -1
#     output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
    
#     output_3D = (output_3D_non_flip + output_3D_flip) / 2
#     return input_2D_non_flip, output_3D

# def setup(opt):
#     torch.manual_seed(opt.seed)
#     torch.cuda.manual_seed_all(opt.seed)
#     np.random.seed(opt.seed)
#     random.seed(opt.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     if opt.train:
#         log_dir = os.path.join(opt.checkpoint, 'train.log')
#         logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=log_dir, level=logging.INFO)
#     return SummaryWriter(log_dir=f'./runs/{opt.model_name}')

# def load_data(opt):
#     dataset_path = os.path.join(opt.root_path, f'data_3d_{opt.dataset}.npz')
#     dataset = Human36mDataset(dataset_path, opt)
#     actions = define_actions(opt.actions)

#     train_loader = None
#     if opt.train:
#         train_dataset = Fusion(opt=opt, train=True, dataset=dataset, root_path=opt.root_path)
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers), pin_memory=True)
        
#     test_dataset = Fusion(opt=opt, train=False, dataset=dataset, root_path=opt.root_path)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
#     return train_loader, test_loader, dataset, actions

# def build_model_and_optimizer(opt, dataset, adj, adj_tem):
#     model = NanoHTNet(opt, adj, adj_tem).cuda()
#     model = nn.DataParallel(model)

#     if opt.reload:
#         model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))[0]
#         print("Reloading model from:", model_path)
#         pre_dict = torch.load(model_path)
#         model_dict = model.state_dict()
        
#         exclude_terms = []
        
#         pre_dict_filtered = {
#             k: v for k, v in pre_dict.items() 
#             if k in model_dict and not any(term in k for term in exclude_terms)
#         }
#         model_dict.update(pre_dict_filtered)
#         model.load_state_dict(model_dict)
    
#     optimizer = optim.Adam(model.parameters(), lr=opt.lr, amsgrad=True)

#     return model, optimizer

# def main(opt):
#     writer = setup(opt)
#     train_loader, test_loader, dataset, actions = load_data(opt)
    
#     adj = adj_mx_from_skeleton(dataset.skeleton())
#     tem_num = opt.keep_frames
#     adj_tem = torch.eye(tem_num, dtype=torch.float, requires_grad=False)
#     for i in range(tem_num):
#         for j in range(tem_num):
#             if abs(i - j) == 1:
#                 adj_tem[i, j] = 1.0
    
#     model, optimizer = build_model_and_optimizer(opt, dataset, adj, adj_tem)
    
#     print("Calculating model complexity...")
#     model_test_for_flops = NanoHTNet(opt, adj, adj_tem).cuda()
#     input_tensor = torch.rand([1, opt.frames, 17, 2]).cuda()
#     flops, params = profile(model_test_for_flops, inputs=(input_tensor,), verbose=False)
#     print(f"FLOPs: {flops / 1e6 * 2:.2f}M, Params: {params / 1e6:.2f}M")
    
#     best_epoch = 0
#     lr = opt.lr

#     for epoch in range(1, opt.nepoch + 1):
#         if opt.train:
#             loss = train(opt, train_loader, model, optimizer)
            
#             # Learning rate decay
#             if epoch % opt.large_decay_epoch == 0:
#                 lr *= opt.lr_decay_large
#             else:
#                 lr *= opt.lr_decay
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
                
#             p1, p2 = val(opt, test_loader, model)
#             writer.add_scalar('finetune/mpjpe', p1, epoch)
#             writer.add_scalar('finetune/pck', p2, epoch)
            
#             is_best = p1 < opt.previous_best
#             log_msg = f"Epoch: {epoch}, LR: {lr:.7f}, Loss: {loss:.4f}, P1: {p1:.2f}, P2: {p2:.2f}, Best: {opt.previous_best:.2f} at epoch {best_epoch}"
            
#             if is_best:
#                 best_epoch = epoch
#                 opt.previous_best = p1
#                 model_name = os.path.join(opt.checkpoint, opt.model_name)
#                 opt.previous_name = save_model(opt.previous_name, model_name, epoch, p1, model)
            
#             print(log_msg)
#             logging.info(log_msg)

#         else: # Evaluation only mode
#             p1, p2 = val(opt, test_loader, model)
#             print(f"Evaluation results -> P1: {p1:.2f}, P2: {p2:.2f}")
#             break

# if __name__ == '__main__':
#     opt = opts().parse()
#     main(opt)



import os
import glob
import math
import torch
import random
import logging
import numpy as np
from tqdm import tqdm

import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torchvision
from ptflops import get_model_complexity_info

from common.opt import opts
from common.utils import *
import common.loss as eval_loss
from common.load_data_h36m import Fusion
from common.h36m_dataset import Human36mDataset

from model.GCN_conv import adj_mx_from_skeleton
from model.trans import HTNet
from model.NanoHTNet import  NanoHTNet

from tensorboardX import SummaryWriter

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

writer = SummaryWriter(log_dir='./runs/' + opt.model_name)


def verify_model_load(model, loaded_state_dict):
    """
    Verifies if a state dictionary was successfully loaded into a model.
    1. Attempts to load with strict=False to report mismatches.
    2. Compares a sample weight before and after loading to confirm changes.
    """
    print("-" * 60)
    print("🚀 Starting Model Load Verification...")

    # --- Step 1: Check a parameter BEFORE loading ---
    # We'll check the last layer's bias as a sample.
    sample_layer_name = list(model.state_dict().keys())[-1]  # e.g., 'module.fc.bias'
    
    try:
        before_load_param = model.state_dict()[sample_layer_name].clone()
        print(f"🔎 Checking parameter: '{sample_layer_name}'")
        # print(f"Value BEFORE load (first 5 elements): {before_load_param.flatten()[:5].cpu().numpy()}")
    except KeyError:
        print(f"❌ Error: Could not find sample layer '{sample_layer_name}' in the current model.")
        print("-" * 60)
        return

    # --- Step 2: Attempt to load the state dictionary ---
    print("\nAttempting to load the state_dict...")
    try:
        # Using strict=False gives us a detailed report instead of an error.
        incompatible_keys = model.load_state_dict(loaded_state_dict, strict=False)
        
        if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
            print("✅ Key Matching: Perfect! All keys from the file matched the model.")
        else:
            if incompatible_keys.missing_keys:
                print(f"⚠️ WARNING: Missing Keys -> These layers are in your model but not in the file:")
                print(f"   {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                print(f"⚠️ WARNING: Unexpected Keys -> These layers are in the file but not in your model:")
                print(f"   {incompatible_keys.unexpected_keys}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during 'load_state_dict': {e}")
        print("-" * 60)
        return

    # --- Step 3: Check the same parameter AFTER loading ---
    after_load_param = model.state_dict()[sample_layer_name]
    # print(f"Value AFTER load (first 5 elements):  {after_load_param.flatten()[:5].cpu().numpy()}")

    # --- Step 4: Compare and conclude ---
    if torch.equal(before_load_param, after_load_param):
        print("\n❌ VERIFICATION FAILED: The sample parameter did NOT change after loading.")
        print("   This strongly suggests the weights were NOT updated from the checkpoint file.")
    else:
        print("\n✅ VERIFICATION SUCCESSFUL: The sample parameter changed, confirming the load.")
    
    print("-" * 60)
    
def train(opt, actions, train_loader, model, optimizer, epoch, loss_ln=None):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model, loss_ln=None):
        with torch.no_grad():
            return step('test', opt, actions, val_loader, model, loss_ln)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None, loss_ln=None):
    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)
    mid_frame = opt.keep_frames//2
    if split == 'train':
        model.train()
    else:
        model.eval()
    
    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        if split =='train':
            output_3D = model(input_2D)
        else:
            input_2D, output_3D = input_augmentation(input_2D, model)


        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0
        if split == 'train':
            index = torch.arange((opt.frames-1)//2-opt.keep_frames//2, (opt.frames-1)//2+opt.keep_frames//2+1)
            out_target = out_target[:,index,:,:]
            loss = mpjpe_cal(output_3D, out_target)
            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        elif split == 'test':
            mid_frame = opt.keep_frames//2
            output_3D = output_3D[:, mid_frame].unsqueeze(1)
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)
        
    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2

def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]
    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip = model(input_2D_flip)
    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip
    return input_2D, output_3D

if __name__ == '__main__':
    manualSeed = opt.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'


    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    tem_num = opt.keep_frames
    adj_tem = torch.eye(tem_num, dtype=torch.float, requires_grad=False)
    for i in range(tem_num):
        for j in range(tem_num):
            if (i==j+1) or (i==j-1):
                adj_tem[i,j] = 1.0



    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                    shuffle=True, num_workers=int(opt.workers), pin_memory=True)           

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path =root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size//opt.stride,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    model = NanoHTNet(opt, adj, adj_tem).cuda()
    model = nn.DataParallel(model)

    if opt.reload:
        model_dict = model.state_dict()
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))[0]
        pre_dict = torch.load(model_path)
        pre_key = pre_dict.keys()

        exclude_terms = []
        for name, key in model_dict.items():
            if name in pre_dict.keys():
                if not any(term in name for term in exclude_terms):
                    model_dict[name] = pre_dict[name]
        model.load_state_dict(model_dict)



    model_test = NanoHTNet(opt, adj, adj_tem).cuda()
    inputs = torch.rand([1, opt.frames, 17, 2]).cuda()
    from thop import profile
    flops, params = profile(model_test.cuda(), inputs=(inputs,))
    flops = flops / 1000000
    params = params / 1000000
    print("FLOPS:"+ str(flops*2))
    print("Param:"+ str(params)+"M")

    print("lr: ", opt.lr)
    print("batch_size: ", opt.batch_size)
    print("channel: ", opt.channel)
    print("GPU: ", opt.gpu)
    print("seed:", opt.seed)
    all_param = []
    lr = opt.lr
    all_param += list(model.parameters())
    optimizer = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.317, patience=5, verbose=True)
    cut = 0

    for epoch in range(1, opt.nepoch):
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)
        p1, p2 = val(opt, actions, test_dataloader, model)
        writer.add_scalar('mpjpe',p1,epoch)
        writer.add_scalar('p2',p1,epoch)

        if opt.train and p1 < opt.previous_best:
            cut = 0
            best_epoch = epoch
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
            opt.previous_best = p1
        if opt.train == 0:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, %d: %.2f, params: %.2f, flops: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best, params, flops))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, %d: %.2f, params: %.2f, flops: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best, params, flops))
        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay
        if cut < 30:
            cut += 1
        else:
            break








