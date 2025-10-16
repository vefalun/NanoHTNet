"""
fuse training and testing

"""

import sys
sys.path.append("..")
import torch.utils.data as data
import numpy as np

from common.camera import *
from common.utils import deterministic_random
from IPython import embed
import common.global_control as global_control


class Fusion_camera(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        if opt.stride == 1:
            from common.generator_pretrain import ChunkedGenerator_pretrain
        else:
            from common.generator_pretrain_stride import ChunkedGenerator_pretrain

        self.opt = opt
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path
        self.pretrain = opt.pretrain

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        # self.rescale = opt.rescale
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        if self.train:
            self.keypoints, self.keypoints_GT = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d, self.poses_train_2d_GT = \
                    self.fetch(dataset, self.train_list, subset=self.subset)
            self.generator = ChunkedGenerator_pretrain(opt.batch_size, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.poses_train_2d_GT, 
                                              self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, 
                                              reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            print('Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints, self.keypoints_GT = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d, self.poses_test_2d_GT = \
                                        self.fetch(dataset, self.test_list, subset=self.subset)
            self.generator = ChunkedGenerator_pretrain(opt.batch_size, self.cameras_test, self.poses_test,
                                              self.poses_test_2d, self.poses_test_2d_GT, self.stride,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            self.key_index = self.generator.saved_index
            print('Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):
        # print('Preparing data...')
        for subject in folder_list:
            # print('load %s' % subject)
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]  
                    # pos_3d[:, 0] = 0

                    if self.keypoints_name.startswith('sh') or self.keypoints_name.startswith('hr'):
                        pos_3d = np.delete(pos_3d,obj=9,axis=1)# remove neck for sh 2D detection

                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        # print('Loading 2D detections...')
        keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',allow_pickle=True)
        keypoints_GT = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + 'gt' + '.npz',allow_pickle=True)
        
        if self.keypoints_name.startswith('sh') or self.keypoints_name.startswith('hr'):
            self.kps_left, self.kps_right = [4,5,6,10,11,12], [1,2,3,13,14,15]
            self.joints_left, self.joints_right = [4,5,6,10,11,12], [1,2,3,13,14,15]
        else:
            keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']

            self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
            self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(
                dataset.skeleton().joints_right())

            # if self.opt.dataset == 'humaneva15':
            #     self.kps_left, self.kps_right = self.joints_left, self.joints_right

        keypoints = keypoints['positions_2d'].item()
        keypoints_GT = keypoints_GT['positions_2d'].item()

        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                for cam_idx in range(len(keypoints[subject][action])):

                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                        keypoints_GT[subject][action][cam_idx] = keypoints_GT[subject][action][cam_idx][:mocap_length]


        # for subject in keypoints_GT.keys():
        for subject in folder_list:
            for action in keypoints_GT[subject]:
                # norm_params = []
                for cam_idx, item in enumerate(keypoints_GT[subject][action]):
                    kps = keypoints[subject][action][cam_idx]
                    kps_GT = keypoints_GT[subject][action][cam_idx]

                    # Normalize camera frame
                    cam = dataset.cameras()[subject][cam_idx]

                    if self.crop_uv == 0:
                            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                            kps_GT[..., :2] = normalize_screen_coordinates(kps_GT[..., :2], w=cam['res_w'], h=cam['res_h'])
                      
                    if self.keypoints_name.startswith('sh') or self.keypoints_name.startswith('hr'):
                        kps_GT = np.delete(kps_GT, obj=9, axis=1) # remove neck for sh 2D detection

                        if self.keypoints_name.startswith('sh'):
                            permute_index = [6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]
                            kps = kps[:, permute_index, :]

                    keypoints[subject][action][cam_idx] = kps
                    keypoints_GT[subject][action][cam_idx] = kps_GT

        return keypoints, keypoints_GT

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        """
        :param dataset:
        :param subjects:
        :param subset:
        :param parse_3d_poses:
        :return: for each pose dict it has key(subject,action,cam_index)
        """
        out_poses_3d = {}
        out_poses_2d = {}
        out_poses_2d_GT = {}
        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]
                poses_2d_GT = self.keypoints_GT[subject][action]

                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d[(subject, action, i)] = poses_2d[i]
                    out_poses_2d_GT[(subject, action, i)] = poses_2d_GT[i]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):  # Iterate across cameras
                        out_poses_3d[(subject, action, i)] = poses_3d[i]


        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                out_poses_2d_GT[key] = out_poses_2d_GT[key][start:start + n_frames:stride]

                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            # Downsample as requested

            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                out_poses_2d_GT[key] = out_poses_2d_GT[key][::stride]

                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_poses_2d_GT

    def __len__(self):
        "Figure our how many sequences we have"
        return len(self.generator.pairs)

    def __getitem__(self, index):
        
        seq_name, start_3d, end_3d, flip, reverse = \
            self.generator.pairs[index], self.generator.bounds_1[index], self.generator.bounds_2[index], \
            self.generator.augment_vectors[index], self.generator.reverse_augment_vectors[index]

        if self.pretrain:
            subject,action,cam_index = seq_name
            seq_str = subject + action + '0'
            is_train = True
            if seq_str in global_control.dict_seq:
                is_train = False
            cam, gt_3D, input_2D, input_2D_GT, input_view1, input_view2, input_view3, input_view4, action, subject, cam_ind, cam_ind_1, cam_ind_2, cam_ind_3,cam_ind_4 = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)
            if self.train == False and self.test_aug:
                _, _, input_2D_aug, input_2D_aug_GT, _, _,_, _, _, _, _, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
                input_2D_GT = np.concatenate((np.expand_dims(input_2D_GT,axis=0),np.expand_dims(input_2D_aug_GT,axis=0)),0)
                
            camera_1 = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804, 1.841107, 4.9552846, 1.5634454], dtype='float32')
            camera_2 = np.array([0.6157188 , -0.7648363 , -0.14833826,0.11794741, 1.7612785,-5.0780067,1.6062651], dtype='float32')
            camera_3 = np.array([0.14651473, -0.14647852, 0.76530236,-0.60941756, -1.8467777,5.2150464,1.4919724], dtype='float32')
            camera_4 = np.array([0.58340085, -0.7853162 , 0.14548823,-0.14749594, -1.7947897,-3.722699,1.5748928], dtype='float32')
            if cam_ind == 0:
                camera_ind = camera_1
            elif cam_ind == 1:
                camera_ind = camera_2
            elif cam_ind == 2:
                camera_ind = camera_3
            elif cam_ind == 3:
                camera_ind = camera_4
            else:
                print("AssertionError")

            input_2D_update = np.concatenate((input_2D, np.expand_dims(np.expand_dims(camera_ind,axis=0).repeat(17,axis=0), axis=0)),axis=2)
            input_2D_update_GT = np.concatenate((input_2D_GT, np.expand_dims(np.expand_dims(camera_ind,axis=0).repeat(17,axis=0), axis=0)),axis=2)
            input_view1 = np.concatenate((input_view1, np.expand_dims(np.expand_dims(camera_1,axis=0).repeat(17,axis=0), axis=0)),axis=2)
            input_view2 = np.concatenate((input_view2, np.expand_dims(np.expand_dims(camera_2,axis=0).repeat(17,axis=0), axis=0)),axis=2)
            input_view3 = np.concatenate((input_view3, np.expand_dims(np.expand_dims(camera_3,axis=0).repeat(17,axis=0), axis=0)),axis=2)
            input_view4 = np.concatenate((input_view4, np.expand_dims(np.expand_dims(camera_4,axis=0).repeat(17,axis=0), axis=0)),axis=2)

            return cam, gt_3D, input_2D_update, input_2D_update_GT, input_view1, input_view2, input_view3, input_view4, action, subject, cam_ind, cam_ind_1, cam_ind_2, cam_ind_3, cam_ind_4
        else:
            cam, gt_3D, input_2D, input_2D_GT, action, subject = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)
            if self.train == False and self.test_aug:
                _, _, input_2D_aug,input_2D_aug_GT, _, _, = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
                input_2D_GT = np.concatenate((np.expand_dims(input_2D_GT,axis=0),np.expand_dims(input_2D_aug_GT,axis=0)),0)
            input_2D_update = input_2D
            input_2D_update_GT = input_2D_GT
            return cam, gt_3D, input_2D_update, input_2D_update_GT, action, subject  