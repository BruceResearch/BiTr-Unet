#Modified from the following:
# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.

# Partially adopted from https://github.com/Issam28/Brain-tumor-segmentation/blob/master/evaluation_metrics.py

from __future__ import absolute_import, print_function
import os
import sys
sys.path.append('./')
import numpy as np
from data_process import load_3d_volume_as_array, binary_dice3d
from scipy import ndimage


def sensitivity(seg,ground): 
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg))
    denom=np.sum(ground)
    if denom==0:
        return 1
    else:
        return  num/denom

def specificity(seg,ground): 
    #computes false positive rate
    num=np.sum(np.multiply(ground==0, seg ==0))
    denom=np.sum(ground==0)
    if denom==0:
        return 1
    else:
        return  num/denom

def sensitivity_whole(seg,ground):
    return sensitivity(seg>0,ground>0)

def sensitivity_en(seg,ground):
    return sensitivity(seg==4,ground==4)

def sensitivity_core(seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return sensitivity(seg_>0,ground_>0)

def specificity_whole(seg,ground):
    return specificity(seg>0,ground>0)

def specificity_en(seg,ground):
    return specificity(seg==4,ground==4)

def specificity_core(seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return specificity(seg_>0,ground_>0)


def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0,0], order=0)
    east = ndimage.shift(binary_map, [1, 0,0], order=0)
    north = ndimage.shift(binary_map, [0, 1,0], order=0)
    south = ndimage.shift(binary_map, [0, -1,0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border


def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg#, border_ref, border_seg

def Hausdorff_distance(ref,seg):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    ref_border_dist, seg_border_dist = border_distance(ref,seg)
    hausdorff_distance = np.max(
        [np.max(ref_border_dist), np.max(seg_border_dist)])
    return hausdorff_distance

def hausdorff_whole (seg,ground):
    return Hausdorff_distance(seg==0,ground==0)

def hausdorff_en (seg,ground):
    return Hausdorff_distance(seg!=4,ground!=4)

def hausdorff_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return Hausdorff_distance(seg_==0,ground_==0)



def get_ground_truth_names(g_folder, patient_names_file):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_gt_names = []
    for patient_name in patient_names:
        patient_dir = os.path.join(g_folder, patient_name)
        img_names   = os.listdir(patient_dir)
        gt_name = None
        for img_name in img_names:
                if 'seg.' in img_name:
                    gt_name = img_name
                    break
        gt_name = os.path.join(patient_dir, gt_name)
        full_gt_names.append(gt_name)
    return full_gt_names

def get_segmentation_names(seg_folder, patient_names_file):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_seg_names = []
    for patient_name in patient_names:
        seg_name = os.path.join(seg_folder, patient_name + '.nii.gz')
        full_seg_names.append(seg_name)
    return full_seg_names

def dice_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    dice_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        dice_one_volume = []
        if(type_idx ==0): #ET
            s_volume[s_volume == 2] = 0
            s_volume[s_volume == 1] = 0 
            g_volume[g_volume == 2] = 0
            g_volume[g_volume == 1] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx == 1): # WT
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx == 2): # TC
            s_volume[s_volume == 2] = 0  
            g_volume[g_volume == 2] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
      
            
        else:
            for label in [1, 2, 4]: # dice of each class
                temp_dice = binary_dice3d(s_volume == label, g_volume == label)
                dice_one_volume.append(temp_dice)
        dice_all_data.append(dice_one_volume)
    return dice_all_data

def sensitivity_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    dice_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        sensi_one_volume = []
        if(type_idx ==0): #ET
            temp_sensi = sensitivity_en(s_volume,g_volume)
            sensi_one_volume = [temp_sensi]
        elif(type_idx == 1): # WT
            temp_sensi = sensitivity_whole(s_volume,g_volume)
            sensi_one_volume = [temp_sensi]
        elif(type_idx == 2): # TC
            temp_sensi = sensitivity_core(s_volume,g_volume)
            sensi_one_volume = [temp_sensi]

    return sensi_one_volume

def specificity_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    dice_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        speci_one_volume = []
        if(type_idx ==0): #ET
            temp_speci = specificity_en(s_volume,g_volume)
            speci_one_volume = [temp_speci]
        elif(type_idx == 1): # WT
            temp_speci = specificity_whole(s_volume,g_volume)
            speci_one_volume = [temp_speci]
        elif(type_idx == 2): # TC
            temp_speci = specificity_core(s_volume,g_volume)
            speci_one_volume = [temp_speci]

    return speci_one_volume

def hd_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    dice_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        hd_one_volume = []
        if(type_idx ==0): #ET
  
            temp_hd = hausdorff_en(s_volume,g_volume)
            hd_one_volume = temp_hd
        elif(type_idx == 1): # WT
            temp_hd = hausdorff_whole(s_volume,g_volume)
            hd_one_volume = temp_hd
        elif(type_idx == 2): # TC

            temp_hd = hausdorff_core(s_volume,g_volume)
            hd_one_volume = temp_hd

    return hd_one_volume


if __name__ == '__main__':
    
    
    
    s_folder = '/Users/qiranjia19961112/Desktop/NYU_RESEARCH/Experiment/Output/ensembleBITRANS'
    g_folder = '/Users/qiranjia19961112/Desktop/NYU_RESEARCH/Experiment/Output/Bi-Trans6000'
    patient_names_file = '/Users/qiranjia19961112/Desktop/NYU_RESEARCH/Experiment/Output/Bi-Trans6000/valid.txt'

    test_types = ['ET','WT', 'TC']
    gt_names  = get_ground_truth_names(g_folder, patient_names_file)
    seg_names = get_segmentation_names(s_folder, patient_names_file)
    for type_idx in range(3):
        dice = dice_of_brats_data_set(gt_names, seg_names, type_idx)
        dice = np.asarray(dice)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/dice_{0:}.txt'.format(test_type), dice)
        np.savetxt(s_folder + '/dice_{0:}_mean.txt'.format(test_type), dice_mean)
        np.savetxt(s_folder + '/dice_{0:}_std.txt'.format(test_type), dice_std)

        '''
        sensitivity = sensitivity_of_brats_data_set(gt_names, seg_names, type_idx)
        np.savetxt(s_folder + '/sensitivity_{0:}.txt'.format(test_type), sensitivity)

        specificity = specificity_of_brats_data_set(gt_names, seg_names, type_idx)
        np.savetxt(s_folder + '/specificity_{0:}.txt'.format(test_type), specificity)
        '''

        hd = hd_of_brats_data_set(gt_names, seg_names, type_idx)


        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 4])
        print('dice mean  ', dice_mean)
        print('dice std   ', dice_std)
        '''
        print('sensitivity  ',sensitivity)
        print('specificity  ',specificity)
        '''
        print('Hausdorff_distance  ',hd)
