import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os
import time

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal_mask(predicted, target, mask):
    assert predicted.shape == target.shape
    # index = [i for i in range(17) if i in mask]
    predicted = predicted[:,:,mask,:]
    target = target[:,:,mask,:]
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1)).contiguous()

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1)).contiguous()

def skeloss(predicted, target):
    assert predicted.shape == target.shape
    start = [0,1,2, 0,4,5, 0,7,8,9,   8,14,15,  8,11,12]
    end =   [1,2,3, 4,5,6, 7,8,9,10, 14,15,16, 11,12,13]
    ske_predicted = torch.zeros(len(start))
    ske_target = torch.zeros(len(start))
    for i in range(len(start)):
        ske_predicted[i] = torch.mean(torch.norm(predicted[:,:,start[i],:] - predicted[:,:,end[i],:], dim=2)).contiguous()
        ske_target[i] = torch.mean(torch.norm(target[:,:,start[i],:] - target[:,:,end[i],:], dim=2)).contiguous()



    return torch.mean(torch.norm(ske_predicted[i] - ske_target[i]))





def frame_loss(predicted):#256,9,17,2
    loss = 0
    for k in range(predicted.size(0)-1):
        for i in range(predicted.size(1)-1):
            for j in range(predicted.size(2)-1):
                loss += (predicted[k+1,i+1,j+1,0] - predicted[k,i,j,0])**2
                loss += (predicted[k+1,i+1,j+1,1] - predicted[k,i,j,1])**2
    return loss

    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1)).contiguous()

## viz loss
def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))




def compute_PCK(gts, preds, scales=1000, eval_joints=None, threshold=150):
    PCK_THRESHOLD = threshold
    sample_num = len(gts)
    total = 0
    true_positive = 0
    if eval_joints is None:
        eval_joints = list(range(gts.shape[1]))

    for n in range(sample_num):
        gt = gts[n]
        pred = preds[n]
        # scale = scales[n]
        scale = 1000
        per_joint_error = np.take(np.sqrt(np.sum(np.power(pred - gt, 2), 1)) * scale, eval_joints, axis=0)
        true_positive += (per_joint_error < PCK_THRESHOLD).sum()
        total += per_joint_error.size

    pck = float(true_positive / total) * 100
    return pck


def compute_AUC(gts, preds, scales=1000, eval_joints=None):
    # This range of thresholds mimics 'mpii_compute_3d_pck.m', which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = np.linspace(0, 150, 31)
    pck_list = []
    for threshold in thresholds:
        pck_list.append(compute_PCK(gts, preds, scales, eval_joints, threshold))

    auc = np.mean(pck_list)

    return auc


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))



# def test_calculation(predicted, target, action, error_sum, data_type, subject):
#     error_sum = mpjpe_by_action(predicted, target, action, error_sum)

#     return error_sum

def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum




def mpjpe_by_action(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name].update(torch.mean(dist).item()*num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name].update(dist[i].item(), 1)

    return action_error_sum


# def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
#     assert predicted.shape == target.shape
#     num = predicted.size(0)
#     dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

#     if len(set(list(action))) == 1:
#         end_index = action[0].find(' ')
#         if end_index != -1:
#             action_name = action[0][:end_index]
#         else:
#             action_name = action[0]

#         action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*num, num)
#     else:
#         for i in range(num):
#             end_index = action[i].find(' ')
#             if end_index != -1:
#                 action_name = action[i][:end_index]
#             else:
#                 action_name = action[i]

#             action_error_sum[action_name]['p1'].update(dist[i].item(), 1)

#     return action_error_sum



def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*batch_num*frame_num, batch_num*frame_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item()*frame_num, frame_num)
            
    return action_error_sum





def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum



def mpjpe_by_joint_mae(predicted, target,num):
    assert predicted.shape == target.shape
    # this is the joint
    mpjpe_joint = torch.mean(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 3),dim=len(target.shape)-4)
    print("\nthe mpjpe/joint",mpjpe_joint)
    # this is the order of joint from big to small
    index = torch.flip(mpjpe_joint.sort(-1).indices,dims=[0])
    index = index.split(num,-1)[0]
    print("\nerror joint",index)
    return index





def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]:
        {'p1':AccumLoss(), 'p2':AccumLoss()}
        for i in range(len(actions))})
    return error_sum

# def define_error_list(actions):
#     error_sum = {}
#     error_sum.update({actions[i]: AccumLoss() for i in range(len(actions))})
#     return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


# def print_error(data_type, action_error_sum, is_train):
#     mean_error = print_error_action(action_error_sum, is_train)

#     return mean_error

# def print_error(data_type, action_error_sum, is_train):
#     mean_error_p1, mean_error_p2 = print_error_action_new(action_error_sum, is_train)

#     return mean_error_p1, mean_error_p2#


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2, pck, auc = 0, 0, 0, 0
    if data_type == 'h36m':
        mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)
        return mean_error_p1, mean_error_p2
    elif data_type.startswith('3dhp'):
        mean_error_p1, mean_error_p2, pck, auc = print_error_action_3dhp(action_error_sum, is_train, data_type)
        return mean_error_p1, mean_error_p2, pck, auc
    elif data_type == 'humaneva15':
        mean_error_p1, mean_error_p2, pck, auc = print_error_action_subject(action_error_sum)
        return mean_error_p1, mean_error_p2, pck, auc



def print_error_action_subject(action_error_sum):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}
    subjects_test = ['Validate/S1', 'Validate/S2', 'Validate/S3']

    print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "Subject", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        for subject in subjects_test:
            print("{0:<12} ".format(action), end="")
            print("{0:<12} ".format(subject.split('/')[-1]), end="")

            for j in range(1, 3):
                mean_error_each['p'+str(j)] = action_error_sum[action][subject]['p'+str(j)].avg * 1000.0
                mean_error_all['p'+str(j)].update(mean_error_each['p'+str(j)], 1)

            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg

def mpjpe_by_joint_new(predicted, target):
    assert predicted.shape == target.shape
    # this is the joint
    mpjpe_joint = torch.mean(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 3),dim=len(target.shape)-4)
    print("\nthe mpjpe/joint",mpjpe_joint)
    # this is the order of joint from big to small
    index = torch.flip(mpjpe_joint.sort(-1).indices,dims=[0])
    # index = index.split(num,-1)[0]
    print("\nerror joint",index)
    return index


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))


    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg



def print_error_action_3dhp(action_error_sum, is_train, data_type):
    mean_error_each = {'p1': 0.0, 'p2': 0.0, 'pck': 0.0, 'auc': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss(), 'auc': AccumLoss()}

    if not is_train:
        if data_type.startswith('3dhp'):
            print("{0:=^12} {1:=^10} {2:=^8} {3:=^8} {4:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK", "AUC"))
        else:
            print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if not is_train:
            print("{0:<12} ".format(action), end="")
            
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        mean_error_each['pck'] = action_error_sum[action]['pck'].avg * 100.0
        mean_error_all['pck'].update(mean_error_each['pck'], 1)

        mean_error_each['auc'] = action_error_sum[action]['auc'].avg * 100.0
        mean_error_all['auc'].update(mean_error_each['auc'], 1)

        if is_train == 0:
            if data_type.startswith('3dhp'):
                print("{0:>6.2f} {1:>10.2f} {2:>10.2f} {3:>10.2f}".format(
                    mean_error_each['p1'], mean_error_each['p2'], 
                    mean_error_each['pck'], mean_error_each['auc']))
            else:
                print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        if data_type.startswith('3dhp'):
            print("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format("Average", 
                mean_error_all['p1'].avg, mean_error_all['p2'].avg,
                mean_error_all['pck'].avg, mean_error_all['auc'].avg))
        else:
            print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                mean_error_all['p2'].avg))

    if data_type.startswith('3dhp'):
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg,  \
                mean_error_all['pck'].avg, mean_error_all['auc'].avg
    else:
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg, 0, 0


def save_model_refine(previous_name, save_dir,epoch, data_threshold, model, model_name):#
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)

    return previous_name


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    return previous_name



def save_model_epoch(previous_name, save_dir, epoch, data_threshold, model):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    return previous_name



def define_actions_humaneva( action ):
  actions = ["Walking", "Jog", "Box"]

  return actions

def define_actions_3dhp( action, train ):
  if train:
    actions = ["Seq1", "Seq2"]
  else:
    actions = ["Seq1"]

    return actions




def gettime():
    time_now = time.localtime()

    month = int(time.strftime("%m", time_now))
    day = int(time.strftime("%d", time_now))
    hour = int(time.strftime("%H", time_now))
    minute = int(time.strftime("%M", time_now))

    return month, day, hour, minute

