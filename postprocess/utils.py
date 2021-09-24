# Modified from https://github.com/MIC-DKFZ/nnunet

import pickle
import torch
import tensorboardX
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk

def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter("./runs/"+model_name.split("/")[-1].split(".h5")[0])

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']
        
        for col in self.header[1:]:
            self.writer.add_scalar(phase+"/"+col, float(values[col]), int(epoch))


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def combine_labels(labels):
    """
    Combine wt, tc, et into WT; tc, et into TC; et into ET
    :param labels: torch.Tensor of size (bs, 3, ?,?,?); ? is the crop size
    :return:
    """
    whole_tumor = labels[:, :3, :, :, :].sum(1)  # could have 2 or 3
    tumor_core = labels[:, 1:3, :, :, :].sum(1)
    enhanced_tumor = labels[:, 2:3, :, :, :].sum(1)
    whole_tumor[whole_tumor != 0] = 1
    tumor_core[tumor_core != 0] = 1
    enhanced_tumor[enhanced_tumor != 0] = 1
    return whole_tumor, tumor_core, enhanced_tumor  # (bs, ?, ?, ?)


def calculate_accuracy(outputs, targets):
    return dice_coefficient(outputs, targets)


def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):  # 搞三个dice看 每个label; 不要做soft dice
    # batch_size = targets.size(0)
    y_pred = outputs[:, :3, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :3, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    wt_truth, tc_truth, et_truth = combine_labels(y_truth)
    res = dict()
    res["dice_wt"] = dice_coefficient_single_label(wt_pred, wt_truth, eps)
    res["dice_tc"] = dice_coefficient_single_label(tc_pred, tc_truth, eps)
    res["dice_et"] = dice_coefficient_single_label(et_pred, et_truth, eps)

    return res


def calculate_accuracy_singleLabel(outputs, targets, threshold=0.5, eps=1e-8):

    y_pred = outputs[:, 0, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, 0, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    res = dice_coefficient_single_label(y_pred, y_truth, eps)
    return res


def dice_coefficient_single_label(y_pred, y_truth, eps):
    # batch_size = y_pred.size(0)
    intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(-3, -2, -1)) + eps / 2  # axis=?, (bs, 1)
    union = torch.sum(y_pred, dim=(-3,-2,-1)) + torch.sum(y_truth, dim=(-3,-2,-1)) + eps  # (bs, 1)
    dice = 2 * intersection / union
    return dice.mean()
    # return dice / batch_size


def load_old_model(model, optimizer, saved_model_path, data_paralell=True):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path, map_location='cpu')
    epoch = checkpoint["epoch"]
    if data_paralell:
        state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():  # remove "module."
            if "module." in k:
                node_name = k[7:]

            else:
                node_name = k
            state_dict[node_name] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, epoch, optimizer


def combine_labels_predicting(output_array):
    """
    # (1, 3, 240, 240, 155)
    :param output_array: output of the model containing 3 seperated labels (3 channels)
    :return: res_array: conbined labels (1 channel)
    """
    shape = output_array.shape[-3:]
    if len(output_array.shape) == 5:
        bs = output_array.shape[0]
        res_array = np.zeros((bs, ) + shape)
        res_array[output_array[:, 0, :, :, :] == 1] = 2  # 1
        res_array[output_array[:, 1, :, :, :] == 1] = 1  # 2
        res_array[output_array[:, 2, :, :, :] == 1] = 4
    elif len(output_array.shape) == 4:
        res_array = np.zeros(shape)
        res_array[output_array[0, :, :, :] == 1] = 2
        res_array[output_array[1, :, :, :] == 1] = 1
        res_array[output_array[2, :, :, :] == 1] = 4
    return res_array


def dim_recovery(img_array, orig_shape=(155, 240, 240)):
    """
    used when doing inference
    :param img_array:
    :param orig_shape:
    :return:
    """
    crop_shape = np.array(img_array.shape[-3:])
    center = np.array(orig_shape) // 2
    lower_limits = center - crop_shape // 2
    upper_limits = center + crop_shape // 2
    if len(img_array.shape) == 5:
        bs, num_labels = img_array.shape[:2]
        res_array = np.zeros((bs, num_labels) + orig_shape)
        res_array[:, :, lower_limits[0]: upper_limits[0],
                        lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array
    if len(img_array.shape) == 4:
        num_labels = img_array.shape[0]
        res_array = np.zeros((num_labels, ) + orig_shape)
        res_array[:, lower_limits[0]: upper_limits[0],
                     lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    if len(img_array.shape) == 3:
        res_array = np.zeros(orig_shape)
        res_array[lower_limits[0]: upper_limits[0],
            lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    return res_array


def convert_stik_to_nparray(gz_path):
    sitkImage = sitk.ReadImage(gz_path)
    nparray = sitk.GetArrayFromImage(sitkImage)
    return nparray


def poly_lr_scheduler(epoch, num_epochs=300, power=0.9):
    return (1 - epoch/num_epochs)**power

