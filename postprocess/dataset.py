# Modified from @github: https://github.com/shu-hai/two-stage-VAE-Attention-gate-BraTS2020

from torch.utils.data import Dataset
import numpy as np
import os
import numpy as np

np.random.seed(0)

import random

random.seed(0)


# from random import sample


def validation_sampling(data_list, test_size=0.2):
    n = len(data_list)
    m = int(n * test_size)
    val_items = random.sample(data_list, m)
    tr_items = list(set(data_list) - set(val_items))
    return tr_items, val_items


def random_intensity_shift(imgs_array, brain_mask, limit=0.1):
    """
    Only do intensity shift on brain voxels
    :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
    :param brain_mask:
    :param limit:
    :return:
    """

    shift_range = 2 * limit
    for i in range(len(imgs_array) - 1):
        factor = -limit + shift_range * np.random.random()
        std = imgs_array[i][brain_mask].std()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] + factor * std
    return imgs_array


def random_scale(imgs_array, brain_mask, scale_limits=(0.9, 1.1)):
    """
    Only do random_scale on brain voxels
    :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
    :param scale_limits:
    :return:
    """
    scale_range = scale_limits[1] - scale_limits[0]
    for i in range(len(imgs_array) - 1):
        factor = scale_limits[0] + scale_range * np.random.random()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] * factor
    return imgs_array


def random_mirror_flip(imgs_array, prob=0.5):
    """
    Perform flip along each axis with the given probability; Do it for all voxels；
    labels should also be flipped along the same axis.
    :param imgs_array:
    :param prob:
    :return:
    """
    for axis in range(1, len(imgs_array.shape)):
        random_num = np.random.random()
        if random_num >= prob:
            if axis == 1:
                imgs_array = imgs_array[:, ::-1, :, :]
            if axis == 2:
                imgs_array = imgs_array[:, :, ::-1, :]
            if axis == 3:
                imgs_array = imgs_array[:, :, :, ::-1]
    return imgs_array


def random_crop(imgs_array, crop_size=(128, 192, 160), lower_limit=(0, 32, 40)):
    """
    crop the image ((155, 240, 240) for brats data) into the crop_size
    the random area is now limited at (0:155, 32:224, 40:200), by default
    :param imgs_array:
    :param crop_size:
    :return:
    """
    orig_shape = np.array(imgs_array.shape[1:])
    crop_shape = np.array(crop_size)
    # ranges = np.array(orig_shape - crop_shape, dtype=np.uint8)
    # lower_limits = np.random.randint(np.array(ranges))
    lower_limit_z = np.random.randint(lower_limit[0], 155 - crop_size[0])
    if crop_size[1] < 192:
        lower_limit_y = np.random.randint(lower_limit[1], 224 - crop_size[1])
    else:
        lower_limit_y = np.random.randint(0, 240 - crop_size[1])
    if crop_size[2] < 160:
        lower_limit_x = np.random.randint(lower_limit[2], 200 - crop_size[2])
    else:
        lower_limit_x = np.random.randint(0, 240 - crop_size[2])
    lower_limits = np.array((lower_limit_z, lower_limit_y, lower_limit_x))
    upper_limits = lower_limits + crop_shape
    imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
                 lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
    return imgs_array


def validation_time_crop(imgs_array, crop_size=(128, 192, 160)):
    """
    crop the image ((155, 240, 240) for brats data) into the crop_size
    :param imgs_array:
    :param crop_size:
    :return:
    """
    orig_shape = np.array(imgs_array.shape[1:])
    crop_shape = np.array(crop_size)
    lower_limit_z = np.random.randint(orig_shape[0] - crop_size[0])
    center_y = 128
    center_x = 120
    lower_limit_y = center_y - crop_size[-2] // 2  # (128, 160, 128)  (?, 48, 56)
    lower_limit_x = center_x - crop_size[-1] // 2  # (128, 192, 160)  (?, 32, 40)
    lower_limits = np.array((lower_limit_z, lower_limit_y, lower_limit_x))

    upper_limits = lower_limits + crop_shape

    imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
                 lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
    return imgs_array


def test_time_crop(imgs_array, crop_size=(144, 192, 160)):
    """
    crop the test image around the center; default crop_zise change from (128, 192, 160) to (144, 192, 160)
    :param imgs_array:
    :param crop_size:
    :return: image with the size of crop_size
    """
    orig_shape = np.array(imgs_array.shape[1:])
    crop_shape = np.array(crop_size)
    center = orig_shape // 2
    lower_limits = center - crop_shape // 2  # (13, 24, 40) (5, 24, 40)
    upper_limits = center + crop_shape // 2  # (141, 216, 200) (149, 216, 200）
    # upper_limits = lower_limits + crop_shape
    imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
                 lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
    return imgs_array


def test_time_flip(imgs_array, tta_idx):
    if tta_idx == 0:  # [0, 0, 0]
        return imgs_array
    if tta_idx == 1:  # [1, 0, 0]
        return imgs_array[:, ::-1, :, :]
    if tta_idx == 2:  # [0, 1, 0]
        return imgs_array[:, :, ::-1, :]
    if tta_idx == 3:  # [0, 0, 1]
        return imgs_array[:, :, :, ::-1]
    if tta_idx == 4:  # [1, 1, 0]
        return imgs_array[:, ::-1, ::-1, :]
    if tta_idx == 5:  # [1, 0, 1]
        return imgs_array[:, ::-1, :, ::-1]
    if tta_idx == 6:  # [0, 1, 1]
        return imgs_array[:, :, ::-1, ::-1]
    if tta_idx == 7:  # [1, 1, 1]
        return imgs_array[:, ::-1, ::-1, ::-1]


def preprocess_label(img, single_label=None):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """

    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET) - orange
    ed = img == 2  # Peritumoral Edema (ED) - yellow
    et = img == 4  # GD-enhancing Tumor (ET) - blue
    if not single_label:
        # return np.array([ncr, ed, et], dtype=np.uint8)
        return np.array([ed, ncr, et], dtype=np.uint8)
    elif single_label == "WT":
        img[ed] = 1
        img[et] = 1
    elif single_label == "TC":
        img[ncr] = 0
        img[ed] = 1
        img[et] = 1
    elif single_label == "ET":
        img[ncr] = 0
        img[ed] = 0
        img[et] = 1
    else:
        raise RuntimeError("the 'single_label' type must be one of WT, TC, ET, and None")
    return img[np.newaxis, :]


class BratsDataset(Dataset):
    def __init__(self, phase, config):
        super(BratsDataset, self).__init__()

        self.config = config
        self.phase = phase
        self.input_shape = config["input_shape"]
        self.data_path = config["data_path"]
        self.seg_label = config["seg_label"]
        self.intensity_shift = config["intensity_shift"]
        self.scale = config["scale"]
        self.flip = config["flip"]

        if phase == "train":
            self.patient_names = config["training_patients"]  # [:4]
        elif phase == "validate" or phase == "evaluation":
            self.patient_names = config["validation_patients"]  # [:2]
        elif phase == "test":
            self.test_path = config["test_path"]
            self.patient_names = config["test_patients"]
            self.tta_idx = config["tta_idx"]

    def __getitem__(self, index):
        patient = self.patient_names[index]
        self.file_path = os.path.join(self.data_path, 'npy', patient + ".npy")
        if self.phase == "test":
            self.file_path = os.path.join(self.test_path, 'npy', patient + ".npy")
        imgs_npy = np.load(self.file_path)

        if self.phase == "train":
            nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
            brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
            for chl in range(len(nonzero_masks)):
                brain_mask = brain_mask | nonzero_masks[chl]  # (155, 240, 240)
            # data augmentation
            cur_image_with_label = imgs_npy.copy()
            cur_image = cur_image_with_label[:-1]
            if self.intensity_shift:
                cur_image = random_intensity_shift(cur_image, brain_mask)
            if self.scale:
                cur_image = random_scale(cur_image, brain_mask)

            cur_image_with_label[:-1] = cur_image
            cur_image_with_label = random_crop(cur_image_with_label, crop_size=self.input_shape[2:])

            if self.flip:  # flip should be performed with labels
                cur_image_with_label = random_mirror_flip(cur_image_with_label)

        elif self.phase == "validate":
            # cur_image_with_label = validation_time_crop(imgs_npy)
            cur_image_with_label = validation_time_crop(imgs_npy, crop_size=self.input_shape[2:])

        elif self.phase == "evaluation":
            cur_image_with_label = imgs_npy.copy()

        if self.phase == "validate" or self.phase == "train" or self.phase == "evaluation":
            inp_data = cur_image_with_label[:-1]
            seg_label = preprocess_label(cur_image_with_label[-1], self.seg_label)
            if self.config["VAE_enable"]:
                final_label = np.concatenate((seg_label, inp_data), axis=0)
            else:
                final_label = seg_label

            return np.array(inp_data), np.array(final_label)

        elif self.phase == "test":
            imgs_npy = test_time_crop(imgs_npy)
            if self.config["predict_from_train_data"]:
                imgs_npy = imgs_npy[:-1]
            imgs_npy = test_time_flip(imgs_npy, self.tta_idx)
            # np.save("../test_time_crop/{}.npy".format(str(index)), imgs_npy)
            # only use when doing inference for training-data
            # imgs_npy = imgs_npy[:4, :, :, :]
            return np.array(imgs_npy)

    # np.array() solve the problem of "ValueError: some of the strides of a given numpy array are negative"

    def __len__(self):
        return len(self.patient_names)