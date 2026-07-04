from random import random
from sklearn.metrics import mean_squared_error
#from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import torch
import numpy as np
from scipy import ndimage
import json

def eval_metrics(output, target):
    metrics = [mse, mean_error, abs_rel_diff, scale_invariant_error, median_error, rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def abs_rel_diff(y_input, y_target, eps = 1e-6):
    abs_diff = np.abs(y_target-y_input)
    return (abs_diff[~np.isnan(abs_diff)]/(y_target[~np.isnan(y_target)]+eps)).mean()

def squ_rel_diff(y_input, y_target, eps = 1e-6):
    abs_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(abs_diff)
    return (abs_diff[~is_nan]**2/(y_target[~is_nan]**2+eps)).mean()

def rms_linear(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(abs_diff)
    return np.sqrt((abs_diff[~is_nan]**2).mean())

def scale_invariant_error(y_input, y_target):
    log_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(log_diff)
    return (log_diff[~is_nan]**2).mean()-(log_diff[~is_nan].mean())**2

def mean_error(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    return abs_diff[~np.isnan(abs_diff)].mean()

def median_error(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    return np.median(abs_diff[~np.isnan(abs_diff)])

def mse(y_input, y_target):
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    sum_mse_over_batch = 0.

    for i in range(N):
        sum_mse_over_batch += mean_squared_error(
            y_input[i, 0, :, :][~np.isnan(y_target[i, 0, :, :])], y_target[i, 0, :, :][~np.isnan(y_target[i, 0, :, :])])

        if C == 3: # color
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 1, :, :][~np.isnan(y_target[i, 1, :, :])], y_target[i, 1, :, :][~np.isnan(y_target[i, 1, :, :])])
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 2, :, :][~np.isnan(y_target[i, 2, :, :])], y_target[i, 2, :, :][~np.isnan(y_target[i, 2, :, :])])

    mean_mse = sum_mse_over_batch / (float(N))
    if C == 3:
        mean_mse /= 3.0

    return mean_mse


def structural_similarity(y_input, y_target):
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    # N x C x H x W -> N x W x H x C -> N x H x W x C
    y_input = np.swapaxes(y_input, 1, 3)
    y_input = np.swapaxes(y_input, 1, 2)
    y_target = np.swapaxes(y_target, 1, 3)
    y_target = np.swapaxes(y_target, 1, 2)
    sum_structural_similarity_over_batch = 0.
    for i in range(N):
        if C == 3:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, :], y_target[i, :, :, :], multichannel=True)
        else:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, 0], y_target[i, :, :, 0])

    return sum_structural_similarity_over_batch / float(N)


class textjson(json.JSONEncoder):
    # 解决不能把tensor写入json
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                return obj.item()
            return obj.tolist()
        return super(textjson, self).default(obj)

#def restore_depth(depth, max_distance=1000, reg_factor=3.7):
def restore_depth(depth, max_distance=80, reg_factor=3.7):
    # 从对数图像恢复到原始深度图
    restored_depth = torch.exp((depth - 1.0) * reg_factor)
    restored_depth = restored_depth * max_distance
    return restored_depth


def abs_rel(pred, gt, eps=1e-6):
    return torch.mean(torch.abs(pred - gt) / (gt + eps))


def rmse_log(pred, gt, eps=1e-6):
    return torch.sqrt(torch.mean((torch.log(pred + eps) - torch.log(gt + eps)) ** 2))

def seed_ev(seed=947):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def si_log(pred, gt, eps=1e-6):
    log_diff = torch.log(pred + eps) - torch.log(gt + eps)
    return torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2


def accuracy(pred, gt, threshold, eps=1e-6):
    return torch.mean((torch.max(pred / (gt + eps), gt / (pred + eps)) < threshold).float())


def mean_absolute_error(pred, gt, cutoff):
    errors = []
    for cutoff_val in cutoff:
        mask = gt < cutoff_val
        # print("mask:",mask)
        errors.append(torch.mean(torch.abs(pred[mask] - gt[mask])))
        # print("errors:",errors)
    return errors