from matplotlib.pyplot import bone
import torch
import numpy as np

def mpjpe(predicted, target, return_joints_err=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if not return_joints_err:
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    else:
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        # errors: [B, T, N]
        from einops import rearrange
        errors = rearrange(errors, 'B T N -> N (B T)')
        errors = torch.mean(errors, dim=-1).cpu().numpy().reshape(-1) * 1000
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1)), errors
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

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
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

def mean_velocity_error(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=axis)
    velocity_target = np.diff(target, axis=axis)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def sym_penalty(dataset,keypoints,pred_out):
    """
    get penalty for the symmetry of human body
    :return:
    """
    loss_sym = 0
    if dataset == 'h36m':
        if keypoints.startswith('hr'):
            left_bone = [(0,4),(4,5),(5,6),(8,10),(10,11),(11,12)]
            right_bone = [(0,1),(1,2),(2,3),(8,13),(13,14),(14,15)]
        else:
            left_bone = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13)]
            right_bone = [(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)]
        for (i_left,j_left),(i_right,j_right) in zip(left_bone,right_bone):
            left_part = pred_out[:,:,i_left]-pred_out[:,:,j_left]
            right_part = pred_out[:, :, i_right] - pred_out[:, :, j_right]
            loss_sym += torch.mean(torch.abs(torch.norm(left_part, dim=-1) - torch.norm(right_part, dim=-1)))
    elif dataset.startswith('STB'):
        loss_sym = 0
    return 0.01*loss_sym

def bonelen_consistency_loss(dataset,keypoints,pred_out):
    loss_length = 0
    if dataset == 'h36m':
        if keypoints.startswith('hr'):
            assert "hrnet has not completed"
        else:
            bones = [(0,1), (0,4), (1,2), (2,3), (4,5), (5,6), (0,7), (7,8), (8,9), (9,10), 
                    (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]
        for (i,j) in bones:
            bonelen = pred_out[:,:,i]-pred_out[:,:,j]
            bone_diff = bonelen[:,1:,:]-bonelen[:,:-1,:]
            loss_length += torch.mean(torch.norm(bone_diff, dim=-1))
    elif dataset.startswith('heva'):
        loss_length = 0

    return 0.01 * loss_length


def temporal_consistency(output_3D, out_target, w_mpjpe):
    dif_seq = output_3D[:,1:,:,:] - output_3D[:,:-1,:,:]
    weights_joints = torch.ones_like(dif_seq).cuda()
    weights_mul = w_mpjpe
    assert weights_mul.shape[0] == weights_joints.shape[-2]
    weights_joints = torch.mul(weights_joints.permute(0,1,3,2),weights_mul).permute(0,1,3,2)
    dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))
    return dif_seq