import numpy as np
from common.opt import opts
opt = opts().parse()


class ChunkedGenerator:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug= False,kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all = False, MAE=False, tds=1):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0

        for key in poses_2d.keys():
            assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            keys = np.tile(np.array(key).reshape([1,3]),(len(bounds - 1),1))
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector,reverse_augment_vector))
            if reverse_aug:
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
            if augment:
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector,~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index,end_index]
            start_index = start_index + poses_3d[key].shape[0]


        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all
        self.MAE = MAE
        self.tds = tds
        self.chunk_length = chunk_length

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        subject,action,cam_index = seq_i
        seq_name = (subject,action,int(cam_index))
        if self.chunk_length == 1:
            start_2d = start_3d - self.pad * self.tds - self.causal_shift
            end_2d = end_3d + self.pad * self.tds - self.causal_shift
        else:
            mid = end_3d - self.pad
            start_2d = mid - self.pad * self.tds - self.causal_shift-1
            end_2d = mid + self.pad * self.tds - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        if opt.pretrain:
            '''
            todo: 需要考虑的点是:
            1.是一个单一的视角作为输入，其他视角作为辅助
            还是随便视角作为输入，其他的视角作为输出
            2. 正样本选择的是一个视角还是一组视角
            '''
            multi_view = [0,1,2,3]
            # multi_view.remove(seq_name[2])


            seq_name_view1 = list(seq_name)
            seq_name_view1[2] = multi_view[0]
            subject_1, action_1, cam_index_1 = seq_name_view1
            seq_view1 = self.poses_2d[tuple(seq_name_view1)].copy()

            seq_name_view2 = list(seq_name)
            seq_name_view2[2] = multi_view[1]
            subject_2, action_2, cam_index_2 = seq_name_view2
            seq_view2 = self.poses_2d[tuple(seq_name_view2)].copy()

            seq_name_view3 = list(seq_name)
            seq_name_view3[2] = multi_view[2]
            subject_3, action_3, cam_index_3 = seq_name_view3
            seq_view3 = self.poses_2d[tuple(seq_name_view3)].copy()

            seq_name_view4 = list(seq_name)
            seq_name_view4[2] = multi_view[3]
            subject_4, action_4, cam_index_4 = seq_name_view4
            seq_view4 = self.poses_2d[tuple(seq_name_view4)].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0:
            data_pad = np.repeat(seq_2d[0:1],pad_left_2d,axis=0)
            new_data = np.concatenate((data_pad, seq_2d[low_2d:high_2d]), axis=0)
            self.batch_2d = new_data[::self.tds]
            if opt.pretrain:
                data_pad_1 = np.repeat(seq_view1[0:1],pad_left_2d,axis=0)
                data_pad_2 = np.repeat(seq_view2[0:1],pad_left_2d,axis=0)
                data_pad_3 = np.repeat(seq_view3[0:1],pad_left_2d,axis=0)
                data_pad_4 = np.repeat(seq_view4[0:1],pad_left_2d,axis=0)
                new_data_1 = np.concatenate((data_pad_1, seq_view1[low_2d:high_2d]), axis=0)
                new_data_2 = np.concatenate((data_pad_2, seq_view2[low_2d:high_2d]), axis=0)
                new_data_3 = np.concatenate((data_pad_3, seq_view3[low_2d:high_2d]), axis=0)
                new_data_4 = np.concatenate((data_pad_4, seq_view4[low_2d:high_2d]), axis=0)
                self.batch_2d_view1 = new_data_1[::self.tds]
                self.batch_2d_view2 = new_data_2[::self.tds]
                self.batch_2d_view3 = new_data_3[::self.tds]
                self.batch_2d_view4 = new_data_4[::self.tds]
                
            #self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')

        elif pad_right_2d != 0:
            data_pad = np.repeat(seq_2d[seq_2d.shape[0]-1:seq_2d.shape[0]], pad_right_2d, axis=0)
            new_data = np.concatenate((seq_2d[low_2d:high_2d], data_pad), axis=0)
            self.batch_2d = new_data[::self.tds]
            if opt.pretrain:
                data_pad_1 = np.repeat(seq_view1[seq_view1.shape[0]-1:seq_view1.shape[0]], pad_right_2d, axis=0)
                data_pad_2 = np.repeat(seq_view2[seq_view2.shape[0]-1:seq_view2.shape[0]], pad_right_2d, axis=0)
                data_pad_3 = np.repeat(seq_view3[seq_view3.shape[0]-1:seq_view3.shape[0]], pad_right_2d, axis=0)
                data_pad_4 = np.repeat(seq_view4[seq_view4.shape[0]-1:seq_view4.shape[0]], pad_right_2d, axis=0)
                new_data_1 = new_data = np.concatenate((seq_view1[low_2d:high_2d], data_pad_1), axis=0)
                new_data_2 = new_data = np.concatenate((seq_view2[low_2d:high_2d], data_pad_2), axis=0)
                new_data_3 = new_data = np.concatenate((seq_view3[low_2d:high_2d], data_pad_3), axis=0)
                new_data_4 = new_data = np.concatenate((seq_view4[low_2d:high_2d], data_pad_4), axis=0)
                self.batch_2d_view1 = new_data_1[::self.tds]
                self.batch_2d_view2 = new_data_2[::self.tds]
                self.batch_2d_view3 = new_data_3[::self.tds]
                self.batch_2d_view4 = new_data_4[::self.tds]
            #self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d:self.tds]
            if opt.pretrain:
                self.batch_2d_view1 = seq_view1[low_2d:high_2d:self.tds]
                self.batch_2d_view2 = seq_view2[low_2d:high_2d:self.tds]
                self.batch_2d_view3 = seq_view3[low_2d:high_2d:self.tds]
                self.batch_2d_view4 = seq_view4[low_2d:high_2d:self.tds]

        if flip:
            self.batch_2d[ :, :, 0] *= -1
            self.batch_2d[ :, self.kps_left + self.kps_right] = self.batch_2d[ :,
                                                                  self.kps_right + self.kps_left]
            if opt.pretrain:
                self.batch_2d_view1[ :, :, 0] *= -1
                self.batch_2d_view1[ :, self.kps_left + self.kps_right] = self.batch_2d_view1[ :,
                                                                    self.kps_right + self.kps_left]         
                self.batch_2d_view2[ :, :, 0] *= -1
                self.batch_2d_view2[ :, self.kps_left + self.kps_right] = self.batch_2d_view2[ :,
                                                                    self.kps_right + self.kps_left]   
                self.batch_2d_view3[ :, :, 0] *= -1
                self.batch_2d_view3[ :, self.kps_left + self.kps_right] = self.batch_2d_view3[ :,
                                                                    self.kps_right + self.kps_left]  
                self.batch_2d_view4[ :, :, 0] *= -1
                self.batch_2d_view4[ :, self.kps_left + self.kps_right] = self.batch_2d_view4[ :,
                                                                    self.kps_right + self.kps_left]  
                                 
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()
            if opt.pretrain:
                self.batch_2d_view1 = self.batch_2d_view1[::-1].copy()
                self.batch_2d_view2 = self.batch_2d_view2[::-1].copy()
                self.batch_2d_view3 = self.batch_2d_view3[::-1].copy()
                self.batch_2d_view4 = self.batch_2d_view4[::-1].copy()

        if not self.MAE:
            if self.poses_3d is not None:
                seq_3d = self.poses_3d[seq_name].copy()
                if self.out_all:
                    low_3d = low_2d
                    high_3d = high_2d
                    pad_left_3d = pad_left_2d
                    pad_right_3d = pad_right_2d
                else:
                    low_3d = max(start_3d, 0)
                    high_3d = min(end_3d, seq_3d.shape[0])
                    pad_left_3d = low_3d - start_3d
                    pad_right_3d = end_3d - high_3d

                if pad_left_3d != 0:
                    data_pad = np.repeat(seq_3d[0:1], pad_left_3d, axis=0)
                    new_data = np.concatenate((data_pad, seq_3d[low_3d:high_3d]), axis=0)
                    self.batch_3d = new_data[::self.tds]
                elif pad_right_3d != 0:
                    data_pad = np.repeat(seq_3d[seq_3d.shape[0] - 1:seq_3d.shape[0]], pad_right_3d, axis=0)
                    new_data = np.concatenate((seq_3d[low_3d:high_3d], data_pad), axis=0)
                    self.batch_3d = new_data[::self.tds]
                    # self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                    #                           ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                else:
                    self.batch_3d = seq_3d[low_3d:high_3d:self.tds]

                if flip:
                    self.batch_3d[ :, :, 0] *= -1
                    self.batch_3d[ :, self.joints_left + self.joints_right] = \
                        self.batch_3d[ :, self.joints_right + self.joints_left]
                if reverse:
                    self.batch_3d = self.batch_3d[::-1].copy()

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[ 2] *= -1
                self.batch_cam[ 7] *= -1

        if self.MAE:
            return self.batch_cam, self.batch_2d.copy(), action, subject, int(cam_index)
        if opt.pretrain:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d, \
                   self.batch_2d_view1, self.batch_2d_view2, self.batch_2d_view3, self.batch_2d_view4,\
                 action, subject, int(cam_index), int(cam_index_1), int(cam_index_2), int(cam_index_3), int(cam_index_4)
        if self.poses_3d is None and self.cameras is None:
            return None, None, self.batch_2d.copy(), action, subject, int(cam_index)
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(),action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, None, self.batch_2d.copy(),action, subject, int(cam_index)
        else:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d.copy(),action, subject, int(cam_index)





            

