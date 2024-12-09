# elastix verison Perform dynamic sequential registrations instantiate elastix and transformix

# Function Perform dynamic sequential registrations instantiate elastix and transformix sandbox
#useful imports
from Ortho_metrics_util import Compute_Orthopedic_Metrics
import pydicom
import tempfile
import easygui
from tqdm import tqdm
import os
import sys
import SimpleITK as sitk
import fnmatch
import shutil
import numpy as np
import pydicom
import errno

import pandas as pd
from transforms3d import transforms_all3d as t3d
from scipy.spatial.transform import Rotation as R
import seaborn as sns

import sys
from os import listdir
import os
import time
import subprocess
import globpip
import shutil
import errno
import time

import sys
import fnmatch
import math
import matplotlib.pyplot as plt
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from sys import platform

import sys
from os import listdir
import os
import time
import subprocess
import glob
import shutil
from multiprocessing import Process, cpu_count
import pydicom
import os

import sys
import fnmatch

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from sys import platform

import os
import glob

import time
import nibabel as nib

# #import matplotlib.pyplot as plt
# import neptune.new as neptune
import torch
# print(torch.__version__)
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn.functional import interpolate
import monai

### Importing functions from monai
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityRanged, CropForegroundd, EnsureType, \
    DeleteItemsd, LoadImaged, RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, ToTensord, RandFlipd, \
    SpatialPadd, RandGaussianNoised, AsChannelFirstd, Orientationd, RandGaussianSmoothd, RandScaleIntensityd, \
    NormalizeIntensityd, AsDiscrete, Rand3DElasticd, RandWeightedCropd, ThresholdIntensityd, EnsureChannelFirstd, \
    RandCropByLabelClassesd, Invertd
from monai.inferers import sliding_window_inference

from monai.data import list_data_collate, DataLoader, CacheDataset, SmartCacheDataset, decollate_batch
from monai.networks.layers import Norm
from monai.networks.nets import DynUNet, SwinUNETR
from monai.losses import DiceLoss, TverskyLoss
from monai.metrics import compute_meandice
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.data.utils import pad_list_data_collate
from monai.data import NibabelReader

# #FOR INFERENCE
import monai
import argparse
import glob
import logging
import os
import shutil
import sys
import torch

import os
import glob

import time

# #import matplotlib.pyplot as plt
# import neptune.new as neptune
import torch

print(torch.__version__)
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import interpolate
import monai

### Importing functions from monai
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityRanged, CropForegroundd, EnsureType, \
    DeleteItemsd, LoadImaged, RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, ToTensord, RandFlipd, \
    SpatialPadd, RandGaussianNoised, AsChannelFirstd, RandGaussianSmoothd, RandScaleIntensityd, NormalizeIntensityd, \
    AsDiscrete, Rand3DElasticd, RandWeightedCropd, MaskIntensityd, ConcatItemsd
from monai.inferers import sliding_window_inference

from monai.data import list_data_collate, DataLoader, CacheDataset, SmartCacheDataset, decollate_batch
from monai.networks.layers import Norm
from monai.networks.nets import DynUNet
from monai.losses import DiceLoss, TverskyLoss
from monai.metrics import compute_meandice
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.data.utils import pad_list_data_collate
from monai.data import NibabelReader
from monai.handlers.utils import from_engine
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    CropForegroundd,
    CastToTyped,
    KeepLargestConnectedComponentd,
    Compose,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandScaleIntensityd,
    RandCropByPosNegLabeld,
    RandGaussianSmoothd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmooth,
    SaveImaged,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    Spacingd,
    Activationsd,
    SpatialPadd,
    ToTensord,
    AddChannel,
    AsChannelFirst,
    ToTensor,
    DivisiblePadd
)

# importing functions from monai

from monai.inferers import sliding_window_inference
from monai.data import list_data_collate
from monai.networks.layers import Norm
from monai.networks.nets import DynUNet
from monai.losses import DiceLoss, TverskyLoss
from monai.metrics import compute_meandice
from monai.utils import set_determinism
import wandb
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import os



# transform point
def transform_points(points, outputpts_path, transform, row, column, mat):
    for t in range(0, len(points)):
        #        trans_inv = np.linalg.inv(transform)
        trans = np.reshape(points[t], (4, 1))
        mat[0, column] = trans[0]
        column = column + 1
        mat[0, column] = trans[1]
        column = column + 1
        mat[0, column] = trans[2]
        column = column + 1

    column = 0
    for t in range(0, len(points)):
        #        trans_inv = np.linalg.inv(transform)
        trans = np.reshape(points[t], (4, 1))
        mat[0,]
        trans[3] = 1
        Tp = transform.dot(trans)

        mat[row, column] = Tp[0]
        column = column + 1
        mat[row, column] = Tp[1]
        column = column + 1
        mat[row, column] = Tp[2]
        column = column + 1

    np.savetxt(outputpts_path, mat, delimiter=",")


def execute_transform_points(path_to_pts, outputpts_path, transform_list):
    # temp = np.loadtxt(path_to_pts, skiprows=1)
    temp = np.loadtxt(path_to_pts)
    points = temp[:, 1:5]
    #                mat = np.zeros((len(transform_list), len(points) * 3))
    mat = np.zeros((len(transform_list) + 1, len(points) * 3))

    row = 1
    for t in transform_list:
        column = 0
        transform_points(points, outputpts_path, np.loadtxt(t), row, column, mat)
        row = row + 1


def execute_transform_points_v2(path_to_pts, outputpts_path, transform_list):
    # temp = np.loadtxt(path_to_pts, skiprows=1)
    temp = path_to_pts
    points = temp[:, 1:5]
    #                mat = np.zeros((len(transform_list), len(points) * 3))
    mat = np.zeros((len(transform_list) + 1, len(points) * 3))

    row = 1
    for t in transform_list:
        column = 0
        transform_points(points, outputpts_path, np.loadtxt(t), row, column, mat)
        row = row + 1


def get_folders(directory, exclude_parts):
    folders = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            if not any(part in item for part in exclude_parts):
                folders.append(os.path.join(directory, item))
    return folders


def get_folders_recursive(directory, exclude_parts):
    folders = []
    for root, dirs, files in os.walk(directory):
        # Exclude folders that contain any of the specified parts in their names
        dirs[:] = [d for d in dirs if not any(part in d for part in exclude_parts)]
        files[:] = [f for f in files if not any(part in f for part in exclude_parts)]
        # Append all remaining directories (folders) to the list\n",
        folders.extend([os.path.join(root, d) for d in dirs])
    return folders

# FUNCtion for creating MHD files from dynamic dicom images
def read_DCM_to_MHD(path_dicom, output_folder):
    '''Path_dicom =   path to the folder containing dynamic dicom data.
       output_folder= path to the folder to hold the MHD files.
    '''
    scanTime = []
    images = []
    images_d = []
    selected_images = []
    temp_directory_list = []

    # create the output folder if it doesnt exist
    path_sorted_dicom = '' + output_folder + '/'
    try:
        os.makedirs(path_sorted_dicom)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # create a temporary folder for intermediate results
    temp_folder = '' + path_sorted_dicom + '/tempfolder' + ''
    try:
        os.makedirs(temp_folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print('EXTRACTING "SCAN TIME" DICOM TAG')
    for image in tqdm(os.listdir(path_dicom)):
        if fnmatch.fnmatch(image, '*'):
            images.append(image)
            input_path = '' + path_dicom + '/' + image + ''
            ds = pydicom.dcmread(input_path, force=True)
            # scanTime.append(float(ds[0x0008, 0x0033].value))  # mid-scan time, this will be used as temporal resolution
            scanTime.append(float(ds[0x0019, 0x1024].value))

    midScanTime = np.unique(scanTime)
    #     midScanTime = np.unique(scanTime)
    timePoints = midScanTime - midScanTime[0]
    # print('midScanTime',midScanTime, 'length',len(midScanTime))
    # print('length',len(midScanTime))
    # print('timePoints',timePoints)
    print('SORTING DICOM IMAGES ACCORDING TO SCAN TIME')
    for image in tqdm(os.listdir(path_dicom)):
        if fnmatch.fnmatch(image, '*'):
            input_path_d = '' + path_dicom + '/' + image + ''
            ds = pydicom.dcmread(input_path_d, force=True)
        for s in midScanTime:
            count = 0
            if float(ds[0x0019, 0x1024].value) == s:

                images_d.append(image)
                SELECTED_input_path = '' + path_dicom + '/' + image + ''
                #                 print("selected input path",SELECTED_input_path)

                scantime = "{:.3f}".format(s)
                time_str = str(scantime)

                time_directory = '' + temp_folder + '/' + time_str + '/'
                temp_directory_list.append(time_directory)
                #         print(time_str)

                try:
                    os.makedirs(time_directory)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise

                shutil.copy(SELECTED_input_path, time_directory)

    i = 0

    sorted_temp_folder = sorted(glob.glob(temp_folder + '/*'))
    print("Writing images:")
    for folders in tqdm(sorted_temp_folder):
        # print('folders',folders)
        #         print("time point:",i)
        output_3d = '' + path_sorted_dicom + '/s01p%02d.mhd' % i
        time_series_data = folders

        #         print("Reading Dicom directory:",  time_series_data)
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(time_series_data)
        #         print("temp directory", time_series_data)
        reader.SetFileNames(dicom_names)

        image = reader.Execute()

        # reader.ReadImageInformation()
        size = image.GetSize()
        #         print("Image size:", size[0], size[1], size[2])

        # print("Writing image:", output_3d)

        sitk.WriteImage(image, output_3d)

        i = i + 1

    try:
        shutil.rmtree(temp_folder)
        # shutil.rmtree(resultDir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    return output_folder


# Helper function to idenfity the fixed image
def Get_fixed_Image(path):
    metric_list = []
    mean_metric_list = []
    time_pt_list = os.listdir(path)

    temp_reg_path = sorted(glob.glob(path + '/*'))

    #     print(temp_reg_path)
    for folder in (temp_reg_path):
        #         print(folder)
        try:
            print(folder + '/IterationInfo.0.R2.txt')
            text = np.genfromtxt(folder + '/IterationInfo.0.R2.txt', skip_header=1)

            for i in range(0, len(text)):
                metric_list.append(text[i][1])

            metric = np.array(metric_list)
            mean_metric = np.mean(metric)

            print('mean metric ', mean_metric)

            mean_metric_list.append(mean_metric)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    print("Fixed image time point:", time_pt_list[mean_metric_list.index(min(mean_metric_list))])
    fixed_timept = time_pt_list[mean_metric_list.index(min(mean_metric_list))]

    return fixed_timept


# Path_to_dynamic_images='/Volumes/Luca_T5/Luca_T5/Clinical_study_knee/Clinical_Study_raw_DICOM_mhd/Healthy/35/Test1/mhd/raw/0_150'
# output='/Volumes/Luca_T5/sandbox_process_4DCT'

# parameter_file_path = '/Volumes/Luca_T5/4D_MSK/Scripts/Param_Files/Euler_Without_Mask_for_preprocessing.txt'
# Static_image_path = '/Volumes/Luca_T5/Luca_T5/Clinical_study_knee/Clinical_Study_raw_DICOM_mhd/Healthy/35/Test1/mhd/raw/Static/s01p01.mhd'
# Static_image = sitk.ReadImage(Static_image_path)
# moving_image_paths=sorted(glob.glob(Path_to_dynamic_images+'/*.mhd'))


## function that runs a registration between the moving images and the static image to identify the fixed image
def runReg_for_fixedImage(Path_to_dynamic_images, parameter_file_path, Static_image_path, Static_image, moving_image,
                          output, count):
    print('Image', moving_image)
    movingImage = sitk.ReadImage(moving_image)
    output_directory = output_directory + '/%02d/' % count

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load images
    moving_image = sitk.ReadImage(moving_image)

    # Set up registration
    registration = sitk.ElastixImageFilter()
    registration.SetFixedImage(Static_image)
    registration.SetMovingImage(moving_image)
    #     registration.SetParameterFileName(parameter_file_path)
    registration.SetParameterMap(registration.ReadParameterFile(parameter_file_path))
    registration.SetOutputDirectory(output_directory)

    # transformixImageFilter.LogToConsoleOff()
    registration.LogToConsoleOff()

    # Perform registration
    registration.Execute()

    # Get registered image
    registered_image = registration.GetResultImage()

    saveoutputfolder = output_directory + '/registered_image.nii'

    # Save registered image
    sitk.WriteImage(registered_image, saveoutputfolder)

def pre_process_images(Path_to_dynamic_images, path_to_output_folder, side, split=False, static=False):
    '''Path_to_dynamic_images = the path to the .MHD files:
       path_to_output_folder = path to the folder where you want your outputs to be
       side= which joint side e.g left or right. this is particular for images such as the thumb, wrist
       split = split tthe images into left and right sides. Particular for images where both limbs will be in the field
       of view. e.g bipedal motion of the knee or ankle

    '''
    print('Path_to_images         = ', Path_to_dynamic_images)
    print('path_to_output_folder  = ', path_to_output_folder)
    print('Joint side             = ', side)
    print('split images           = ', split)
    image_paths = sorted(glob.glob(Path_to_dynamic_images + '/*.mhd'))

    if split:

        print('Images will be split into "Left and Right" sides')
        count = 0
        if static:
            print('Processing static image')
            static_path = glob.glob(Path_to_dynamic_images + '/*.mhd')
            # print('static path',static_path[0])
            fixedImage = sitk.ReadImage(static_path[0])

            mid = fixedImage.GetSize()[0] / 2
            # split images into left and right
            right = fixedImage[0:int(mid), 0:fixedImage.GetSize()[1], 0:fixedImage.GetSize()[2]]
            left = fixedImage[int(mid) + 1:fixedImage.GetSize()[0], 0:fixedImage.GetSize()[1],
                   0:fixedImage.GetSize()[2]]

            # sitk.WriteImage(fixedImage,'/Volumes/Luca_T5/sandbox_process_4DCT/Static/right.mhd')

            otsuImage_R = sitk.OtsuMultipleThresholds(right, numberOfThresholds=2)
            otsuImage_L = sitk.OtsuMultipleThresholds(left, numberOfThresholds=2)

            threshold_R = sitk.BinaryThreshold(otsuImage_R, 2, 2)
            connected_R = sitk.ConnectedComponent(threshold_R)
            labelConnected_R = sitk.RelabelComponent(connected_R)

            threshold_L = sitk.BinaryThreshold(otsuImage_L, 2, 2)
            connected_L = sitk.ConnectedComponent(threshold_L)
            labelConnected_L = sitk.RelabelComponent(connected_L)

            # select right leg
            thresholdlabelConnected_right = sitk.BinaryThreshold(labelConnected_R, 1, 1)

            # select left leg
            thresholdlabelConnected_left = sitk.BinaryThreshold(labelConnected_L, 1, 1)
            # sitk.WriteImage(thresholdlabelConnected_right,'/Volumes/Luca_T5/sandbox_process_4DCT/Static/thresholdlabelConnected_left.mhd')

            binaryholeFill_right = sitk.BinaryMorphologicalClosing(thresholdlabelConnected_right, (10, 10, 10))

            binaryholeFill_left = sitk.BinaryMorphologicalClosing(thresholdlabelConnected_left, (10, 10, 10))
            finalResults_right = sitk.Mask(right, binaryholeFill_right, -1028)
            finalResults_left = sitk.Mask(left, binaryholeFill_left, -1028)

            R = (path_to_output_folder + "/right")
            L = (path_to_output_folder + "/left")

            # print('R',path_to_output_folder+"Static/right")
            try:
                os.makedirs(R)
                os.makedirs(L)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            # sitk.WriteImage(thresholdlabelConnected_right,"/Users/bkeelson/Data/Knee/Input/01/Fixed/s01p10.mhd")
            outputpath_R = (R + "/s01p%02d.mhd") % (count)
            outputpath_L = (L + "/s01p%02d.mhd") % (count)
            # count = count + 1

            sitk.WriteImage(finalResults_right, outputpath_R)
            sitk.WriteImage(finalResults_left, outputpath_L)

            if os.path.isfile(static_path[0]):
                os.remove(static_path[0])
            else:
                # If it fails, inform the user.
                print("Error: %s file not found" % static_path[0])

            return outputpath_R, outputpath_L



        else:
            print('Processing dynamic images')
            R = (path_to_output_folder + "/right")
            L = (path_to_output_folder + "/left")

            try:
                os.makedirs(R)
                os.makedirs(L)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
            count = 0

            for image in tqdm(image_paths):
                # print('Image', image)
                fixedImage = sitk.ReadImage(image)
                mid = fixedImage.GetSize()[0] / 2
                # split images into left and right
                right = fixedImage[0:int(mid), 0:fixedImage.GetSize()[1], 0:fixedImage.GetSize()[2]]
                left = fixedImage[int(mid) + 1:fixedImage.GetSize()[0], 0:fixedImage.GetSize()[1],
                       0:fixedImage.GetSize()[2]]

                otsuImage_R = sitk.OtsuMultipleThresholds(right, numberOfThresholds=2)
                otsuImage_L = sitk.OtsuMultipleThresholds(left, numberOfThresholds=2)

                threshold_R = sitk.BinaryThreshold(otsuImage_R, 2, 2)
                connected_R = sitk.ConnectedComponent(threshold_R)
                labelConnected_R = sitk.RelabelComponent(connected_R)

                threshold_L = sitk.BinaryThreshold(otsuImage_L, 2, 2)
                connected_L = sitk.ConnectedComponent(threshold_L)
                labelConnected_L = sitk.RelabelComponent(connected_L)

                # select right leg
                thresholdlabelConnected_right = sitk.BinaryThreshold(labelConnected_R, 1, 1)

                # select left leg
                thresholdlabelConnected_left = sitk.BinaryThreshold(labelConnected_L, 1, 1)

                binaryholeFill_right = sitk.BinaryMorphologicalClosing(thresholdlabelConnected_right, (10, 10, 10))

                binaryholeFill_left = sitk.BinaryMorphologicalClosing(thresholdlabelConnected_left, (10, 10, 10))
                finalResults_right = sitk.Mask(right, binaryholeFill_right, -1028)
                finalResults_left = sitk.Mask(left, binaryholeFill_left, -1028)

                # sitk.WriteImage(thresholdlabelConnected_right,"/Users/bkeelson/Data/Knee/Input/01/Fixed/s01p10.mhd")
                outputpath_R = (R + "/s01p%02d.mhd") % (count)
                outputpath_L = (L + "/s01p%02d.mhd") % (count)
                count = count + 1

                sitk.WriteImage(finalResults_right, outputpath_R)
                sitk.WriteImage(finalResults_left, outputpath_L)

                if os.path.isfile(image):
                    os.remove(image)
                else:
                    # If it fails, inform the user.
                    print("Error: %s file not found" % image)

            return R, L

    else:
        count = 0
        for image in tqdm(image_paths):
            # print('Image', image)
            fixedImage = sitk.ReadImage(image)
            otsuImage_R = sitk.OtsuMultipleThresholds(fixedImage, numberOfThresholds=2)
            threshold_R = sitk.BinaryThreshold(otsuImage_R, 2, 2)
            connected_R = sitk.ConnectedComponent(threshold_R)
            labelConnected_R = sitk.RelabelComponent(connected_R)
            thresholdlabelConnected_right = sitk.BinaryThreshold(labelConnected_R, 1, 1)
            binaryholeFill_right = sitk.BinaryMorphologicalClosing(thresholdlabelConnected_right, (10, 10, 10))
            finalResults_right = sitk.Mask(fixedImage, binaryholeFill_right, -1028)

            R = (path_to_output_folder + "/" + side)
            try:
                os.makedirs(R)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            # sitk.WriteImage(thresholdlabelConnected_right,"/Users/bkeelson/Data/Knee/Input/01/Fixed/s01p10.mhd")
            outputpath_R = (R + "/s01p%02d.mhd") % (count)
            count = count + 1

            sitk.WriteImage(finalResults_right, outputpath_R)
            if os.path.isfile(image):
                os.remove(image)
            else:
                # If it fails, inform the user.
                print("Error: %s file not found" % image)

        # clean up

        return R, _


def roundThousand(x):
    y = int(1000.0 * x + 0.5)
    return str(float(y) * .001)


def elapsedTime(start_time):
    dt = roundThousand(time.perf_counter() - start_time)
    print("    ", dt, "seconds")


def Segmentation_2_Mesh(sitk_image, smoothness, outputpath_stl, outputpath_ply):
    #     try:
    #         os.makedirs(outputpath, exist_ok = True)

    #     except OSError as error:
    #         print("Directory '%s' can not be created")

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    vtkimg = sitk2vtk.sitk2vtk(sitk_image)
    mesh = vtkutils.extractSurface(vtkimg, 1)
    mesh2 = vtkutils.cleanMesh(mesh, False)
    mesh3 = vtkutils.smoothMesh(mesh2, smoothness)
    vtkutils.writeMesh(mesh3, outputpath_stl)
    vtkutils.writeMesh(mesh3, outputpath_ply)


def transform_label(label_image, general_path, output_folder, transformParameterFile, STL, fixed_image, smoothness=50):
    t = time.perf_counter()
    if isinstance(label_image, str):
        label_image = sitk.ReadImage(label_image)

    rigid_inverse = general_path + '/Scripts/Param_Files/Euler_Without_Mask_INVERSE.txt'

    transformixImageFilter = sitk.TransformixImageFilter()

    parameterMap = sitk.ReadParameterFile(transformParameterFile)
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)

    # Set up inverse registration
    registration_inverse = sitk.ElastixImageFilter()
    registration_inverse.SetFixedImage(fixed_image)
    registration_inverse.SetMovingImage(fixed_image)
    registration_inverse.SetInitialTransformParameterFileName(transformParameterFile)
    registration_inverse.SetParameterMap(registration_inverse.ReadParameterFile(rigid_inverse))
    registration_inverse.LogToConsoleOff()

    # inverse
    registration_inverse.Execute()
    parameterMap_inverse = registration_inverse.GetTransformParameterMap()
    parameterMap_inverse[0]['InitialTransformParametersFileName'] = ['NoInitialTransform']
    parameterMap_inverse[0]['FinalBSplineInterpolationOrder'] = ['0']
    parameterMap_inverse[0]['DefaultPixelValue'] = ['0']
    parameterMap_inverse[0]['ResultImagePixelType'] = ['unsigned char']

    ## transform label using the inverse
    transformixImageFilter.SetTransformParameterMap(parameterMap_inverse)
    transformixImageFilter.SetMovingImage(label_image)
    transformixImageFilter.SetOutputDirectory(output_folder)
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.Execute()

    transformed_label = transformixImageFilter.GetResultImage()

    # cast to unchar
    mask = sitk.Cast(transformed_label, sitk.sitkUInt8)
    # transformed_labels.append(mask)
    timept = output_folder.split('/')[-1]
    bone = output_folder.split('/')[-2].split('_')[0]
    STL = STL + '/' + bone
    transformed_mask = output_folder + '/transformed_mask_' + timept + '.nii'

    try:
        os.makedirs(STL, exist_ok=True)

    except OSError as error:
        print("Directory '%s' can not be created")

    outputpath_stl = STL + '/transformed_mask_' + timept + '.stl'
    outputpath_ply = STL + '/transformed_mask_' + timept + '.ply'
    outputpath_obj = STL + '/transformed_mask_' + timept + '.obj'

    # if debug: save result for each transformed label
    sitk.WriteImage(mask, transformed_mask)
    Segmentation_2_Mesh(mask, smoothness, outputpath_stl)
    Segmentation_2_Mesh(mask, smoothness, outputpath_ply)
    convert_stl_to_obj(outputpath_stl, outputpath_obj)

    elapsedTime(t)

    ##edited to return an array matrix which contains the sequential 4x4 transformations for each bone's time point
    # example the concantenated 4x4 matrix for mc1 at time point 5 will be in position
    # array_of_matrices[0,5] with 0 implying results for mc1 are stored first


def Calculate_cardan_angles_Knee(path_reg, fixed_pos, mask_names, sequence='zyx', init_tibia=False,
                                 relative_motion=True):
    print('Computation using ', sequence, ' sequence')

    def compute_relative_motion(matrix_A, matrix_B):

        """
        Compute the 4x4 matrix defining the motion of B relative to A.
        Parameters:
        - A: 4x4 numpy array representing the initial transformation matrix (Stationary or reference )
        - B: 4x4 numpy array representing the final transformation matrix (moving)

        Returns:
        - Relative_Matrix: 4x4 numpy array defining the motion of B relative to A
        """
        # Check if the input matrices are 4x4
        if matrix_A.shape != (4, 4) or matrix_B.shape != (4, 4):
            raise ValueError("Input matrices must be 4x4 transformation matrices")

        # Compute the relative transformation matrix
        relative_motion_matrix = np.dot(matrix_B, np.linalg.inv(matrix_A))
        return relative_motion_matrix

    def save_Cardan_angles(angles, txtfilename, csvfilename, time):

        np.savetxt(txtfilename, angles)
        dataframe = pd.read_csv(txtfilename, delimiter=' ')
        dataframe.columns = ['X', 'Y', 'Z']
        dataframe['Time'] = time.tolist()

        # storing this dataframe in a csv file
        dataframe.to_csv(csvfilename,
                         index=None)

    def eulerAnglesToTransformationMatrix(trans, sequence):

        # (TransformParameters θ_x , θ_y , θ_z , t_x , t_y , t_z)

        R_x = np.array([[1, 0, 0, 0],
                        [0, math.cos(trans[0]), -math.sin(trans[0]), 0],
                        [0, math.sin(trans[0]), math.cos(trans[0]), 0],
                        [0, 0, 0, 1]
                        ])

        R_y = np.array([[math.cos(trans[1]), 0, math.sin(trans[1]), 0],
                        [0, 1, 0, 0],
                        [-math.sin(trans[1]), 0, math.cos(trans[1]), 0],
                        [0, 0, 0, 1]
                        ])

        R_z = np.array([[math.cos(trans[2]), -math.sin(trans[2]), 0, 0],
                        [math.sin(trans[2]), math.cos(trans[2]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ])

        T = np.array([[1, 0, 0, trans[3]],
                      [0, 1, 0, trans[4]],
                      [0, 0, 1, trans[5]],
                      [0, 0, 0, 1]
                      ])
        if sequence == 'xyz':
            R = R_x @ R_y @ R_z
            # print('Computation using ', sequence, ' sequence')
        if sequence == 'zyx':
            R = R_z @ R_y @ R_x
            # print('Computation using ', sequence, ' sequence')
        if sequence == 'zxy':
            R = R_z @ R_x @ R_y
            # print('Computation using ', sequence, ' sequence')
        # else:
        #     print('Unknown sequence given')

        M = R @ T
        return R, M, T

    def isRotationMatrix(R):
        # Checks if a matrix is a valid rotation matrix.
        # Used in rotationMatrixToEulerAngles
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(R):
        # Calculates rotation matrix from euler Angles
        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def PathToEulerParameters(transformPath):
        # Retrieves arrays with parameters from the TransformParameter in transformPath
        transformPMap = sitk.ElastixImageFilter().ReadParameterFile(transformPath)
        transform = np.asarray([float(i) for i in transformPMap['TransformParameters']])
        center = np.asarray([float(i) for i in transformPMap['CenterOfRotationPoint']])
        origin = np.asarray([float(i) for i in transformPMap['Origin']])
        return transform, center, origin

    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    def PathToTx(transformPath, sequence):
        # Converts the euler parameters into a transformation matrix using the
        # formula Tx= R(x-C)+T+C => Tx = -Rc+T+c

        (eulertransform, center, origin) = PathToEulerParameters(transformPath)  # obtain Euler parameters
        (RotMatrix, _, TransMatrix) = eulerAnglesToTransformationMatrix(eulertransform,
                                                                        sequence)  # obtain matrices from euler
        center = np.append(center, 1)  # center now has size (4,1), to operate
        newTransMatrix = np.identity(4, float)
        newTransMatrix[:, 3] = -1 * RotMatrix @ center + TransMatrix[:, 3] + center
        Tx = newTransMatrix @ RotMatrix

        return Tx

    # Transparam = np.zeros((39, 6))
    pos_selector = 0

    # for subject in Subject:
    print('FIXED POSITION', fixed_pos[pos_selector], 'for file ', path_reg)

    moving_image_paths = sorted(glob.glob(f'{path_reg}/Moving/*.mhd'))
    #moving_image_paths = sorted(glob.glob(f'{path_reg}/Output/{mask_names[0]}/*'))
    print('moving_image_paths', f'{path_reg}/Moving/*.mhd')
    directory = f'{path_reg}/Output/{mask_names[0]}'
    print('directory', directory)
    exclude_parts = ['Transform']
    timept_folders = get_folders(directory, exclude_parts)
    print('no timpt_folders', len(timept_folders))

    num_moving_images = len(moving_image_paths)

    m = len(mask_names)  # Number of bones or structures to be studied
    n = num_moving_images  # Number of time points in each structure

    # Create a list of m arrays, each containing n 4x4 identity matrices
    list_of_arrays = [
        [
            np.eye(4)  # Use np.eye to create a 4x4 identity matrix
            for _ in range(n)
        ]
        for _ in range(m)
    ]

    # Convert the list to a NumPy array
    array_of_matrices = np.array(list_of_arrays)

    # if relative_motion==False:\n",
    print('computing cardan angles')

    fixed_pos=[0]
    for bone_count, bone in enumerate(mask_names):

        # create folder to hold transformation matrices\n",
        outdirname = f'{path_reg}/Output/{bone}/Transform'
        try:
            os.makedirs(outdirname)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        outdirname_r = f'{path_reg}/Output/Relative_Transform'
        try:
            os.makedirs(outdirname_r)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        outdirname_ref = f'{path_reg}/Output/Relative_Transform_ref'
        try:
            os.makedirs(outdirname_ref)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        cardan_angles = np.zeros((num_moving_images, 3), dtype=float)
        # print(\"Bone\", bone, \"bone count\",bone_count )
        # timepts = f'{path_reg}/{subject}/{Test}/{Leg}/{motion}/{bone}\n",

        im_no = num_moving_images

        print('im_no', im_no)
        subjTx = [np.identity(4, float)] * im_no
        init_Tx = [np.identity(4, float)] * im_no
        subjTx_comb = [np.identity(4, float)] * im_no
        subjEuler = [0] * im_no
        newEuler = [0] * im_no
        centerTx = [0] * im_no
        Tx_relative = [np.identity(4, float)] * im_no
        combinedMatrix = [np.identity(4, float)] * im_no

        combinedMatrix_tib = [np.identity(4, float)] * im_no
        combinedMatrix_pat = [np.identity(4, float)] * im_no

        # #lists to store the sequential transformation matrices i.e the concatenated matrices
        # Sequential_Tx=[]
        # Sequential_Tx_relative=[]

        if init_tibia and 'Patella' in bone:

            for j in range(0, im_no):
                # print('j',j)\n",

                transformPath = f'{path_reg}/Output/{bone}/{(j):02d}/TransformParameters.0.txt'

                transformPath_init = f'{path_reg}/Output/{mask_names[2]}/{(j):02d}/TransformParameters.0.txt'
                subjTx[j] = PathToTx(transformPath, sequence)  # retrieve uncombined transformation matrices\n",
                init_Tx[j] = PathToTx(transformPath_init, sequence)
                # Tx1 = Compose4x4matrix(a1, b1, c1)
                combinedMatrix_pat[j] = subjTx[j].dot(combinedMatrix_pat[j - 1])
                combinedMatrix_tib[j] = init_Tx[j].dot(combinedMatrix_tib[j - 1])
                # combinedMatrix = subjTx[j].dot(Tx_tib_combined)

            subjTx_comb[fixed_pos[pos_selector]] = subjTx[fixed_pos[pos_selector]]
            np.savetxt(f'{path_reg}/Output/{bone}/Transform/{(fixed_pos[pos_selector]):02d}_tx.txt',
                       subjTx[fixed_pos[pos_selector]])

            # # Succesive matrix combination, starting with fixed frame and going

            # # upwards and downwards.

            for up in range(fixed_pos[pos_selector] + 1, im_no, 1):
                subjTx_comb[up] = (subjTx[up].dot(combinedMatrix_tib[up]))

                array_of_matrices[bone_count, up] = subjTx_comb[up]
                np.savetxt(f'{path_reg}/Output/{bone}/Transform/{(up):02d}_tx_v5.txt', subjTx_comb[up])

                translation_combined = t3d.translation_from_matrix(subjTx_comb[up])
                angles_combined = t3d.euler_from_matrix(subjTx_comb[up], axes='s' + sequence)
                # Create a Rotation object from the rotation part of the transformation matrix
                rotation = R.from_matrix(subjTx_comb[up][:3, :3])

                # Get the Euler angles using a specific order
                euler_angles = rotation.as_euler(sequence, degrees=True)

                # print('angles_combined', angles_combined[0] * 57.295779513, angles_combined[1] * 57.295779513,
                #       angles_combined[2] * 57.295779513)
                #            print('translation_combined',translation_combined)
                cardan_angles[up][2] = angles_combined[2] * 57.295779513
                cardan_angles[up][1] = angles_combined[1] * 57.295779513
                cardan_angles[up][0] = angles_combined[0] * 57.295779513

                cardan_angles[up][2] = euler_angles[2]
                cardan_angles[up][1] = euler_angles[1]
                cardan_angles[up][0] = euler_angles[0]

            for down in range(fixed_pos[pos_selector] - 1, 0, -1):
                subjTx_comb[down] = (subjTx[down].dot(combinedMatrix_tib[down]))
                array_of_matrices[bone_count, down] = subjTx_comb[down]
                np.savetxt(f'{path_reg}/Output/{bone}/Transform/{(down):02d}_tx.txt', subjTx_comb[down])

                translation_combined = t3d.translation_from_matrix(subjTx_comb[down])
                angles_combined = t3d.euler_from_matrix(subjTx_comb[down], axes='s' + sequence)
                # Create a Rotation object from the rotation part of the transformation matrix
                rotation = R.from_matrix(subjTx_comb[down][:3, :3])

                # Get the Euler angles using a specific order
                euler_angles = rotation.as_euler(sequence, degrees=True)

                # print('angles_combined', angles_combined[0] * 57.295779513, angles_combined[1] * 57.295779513,
                #       angles_combined[2] * 57.295779513)
                #            print('translation_combined',translation_combined)
                cardan_angles[down][2] = angles_combined[2] * 57.295779513
                cardan_angles[down][1] = angles_combined[1] * 57.295779513
                cardan_angles[down][0] = angles_combined[0] * 57.295779513

                cardan_angles[down][2] = euler_angles[2]
                cardan_angles[down][1] = euler_angles[1]
                cardan_angles[down][0] = euler_angles[0]

            # save cardan angles to txt and csv files
            txtfilename = f'{path_reg}/Output/{bone}/Cardan_angles.txt'
            csvfilename = f'{path_reg}/Output/{bone}/Cardan_angles.csv'
            excelfilename = f'{path_reg}/Output/{bone}/Cardan_angles.xlsx'
            time = np.arange(0, im_no - 1)

            save_Cardan_angles(cardan_angles, txtfilename, csvfilename, time)
            read_file = pd.read_csv(csvfilename)
            read_file.to_excel(excelfilename, index=None, header=True)

        else:
            for j in range(0, im_no):
                # print('j',j)

                transformPath = f'{path_reg}/Output/{bone}/{(j):02d}/TransformParameters.0.txt'
                subjTx[j] = PathToTx(transformPath, sequence)  # retrieve uncombined transformation matrices
                Tx1 = PathToTx(transformPath, sequence)
                # Tx1 = Compose4x4matrix(a1, b1, c1)
                # combinedMatrix = Tx1.dot(combinedMatrix)

            subjTx_comb[fixed_pos[pos_selector]] = subjTx[fixed_pos[pos_selector]]
            np.savetxt(f'{path_reg}/Output/{bone}/Transform/{(fixed_pos[pos_selector]):02d}_tx.txt',
                       subjTx[fixed_pos[pos_selector]])

            # # Succesive matrix combination, starting with fixed frame and going

            # # upwards and downwards.

            for up in range(fixed_pos[pos_selector] + 1, im_no, 1):
                subjTx_comb[up] = subjTx[up].dot(subjTx_comb[up - 1])
                array_of_matrices[bone_count, up] = subjTx_comb[up]
                np.savetxt(f'{path_reg}/Output/{bone}/Transform/{(up):02d}_tx.txt', subjTx_comb[up])

                translation_combined = t3d.translation_from_matrix(subjTx_comb[up])
                angles_combined = t3d.euler_from_matrix(subjTx_comb[up], axes='s' + sequence)
                # Create a Rotation object from the rotation part of the transformation matrix
                rotation = R.from_matrix(subjTx_comb[up][:3, :3])

                # Get the Euler angles using a specific order
                euler_angles = rotation.as_euler(sequence, degrees=True)

                # print('angles_combined', angles_combined[0] * 57.295779513, angles_combined[1] * 57.295779513,\n",
                #       angles_combined[2] * 57.295779513)
                #            print('translation_combined',translation_combined)
                cardan_angles[up][2] = angles_combined[2] * 57.295779513
                cardan_angles[up][1] = angles_combined[1] * 57.295779513
                cardan_angles[up][0] = angles_combined[0] * 57.295779513

                cardan_angles[up][2] = euler_angles[2]
                cardan_angles[up][1] = euler_angles[1]
                cardan_angles[up][0] = euler_angles[0]

            for down in range(fixed_pos[pos_selector] - 1, 0, -1):
                subjTx_comb[down] = subjTx[down].dot(subjTx_comb[down + 1])
                array_of_matrices[bone_count, down] = subjTx_comb[down]
                np.savetxt(f'{path_reg}/Output/{bone}/Transform/{(down):02d}_tx.txt', subjTx_comb[down])

                translation_combined = t3d.translation_from_matrix(subjTx_comb[down])
                angles_combined = t3d.euler_from_matrix(subjTx_comb[down], axes='s' + sequence)
                # Create a Rotation object from the rotation part of the transformation matrix\n",
                rotation = R.from_matrix(subjTx_comb[down][:3, :3])

                # Get the Euler angles using a specific order
                euler_angles = rotation.as_euler(sequence, degrees=True)

                # print('angles_combined', angles_combined[0] * 57.295779513, angles_combined[1] * 57.295779513,
                #       angles_combined[2] * 57.295779513)
                #            print('translation_combined',translation_combined)
                cardan_angles[down][2] = angles_combined[2] * 57.295779513
                cardan_angles[down][1] = angles_combined[1] * 57.295779513
                cardan_angles[down][0] = angles_combined[0] * 57.295779513

                cardan_angles[down][2] = euler_angles[2]
                cardan_angles[down][1] = euler_angles[1]
                cardan_angles[down][0] = euler_angles[0]

            # save cardan angles to txt and csv files
            txtfilename = f'{path_reg}/Output/{bone}/Cardan_angles.txt'
            csvfilename = f'{path_reg}/Output/{bone}/Cardan_angles.csv'
            excelfilename = f'{path_reg}/Output/{bone}/Cardan_angles.xlsx'
            time = np.arange(0, im_no - 1)

            save_Cardan_angles(cardan_angles, txtfilename, csvfilename, time)
            read_file = pd.read_csv(csvfilename)
            read_file.to_excel(excelfilename, index=None, header=True)

    print('computing relative motion cardan angles')
    ref_bone = mask_names[0]

    cardan_angles = np.zeros((num_moving_images, 3), dtype=float)
    euler_angles_res = np.zeros((num_moving_images, 3), dtype=float)
    im_no = num_moving_images

    subjTx_ref = [np.identity(4, float)] * im_no
    subjTx_target = [np.identity(4, float)] * im_no
    subjTx_comb_ref = [np.identity(4, float)] * im_no
    subjTx_comb_target = [np.identity(4, float)] * im_no
    Tx_relative = [np.identity(4, float)] * im_no
    Tx_relative_ref = [np.identity(4, float)] * im_no
    combinedRelativeMatrix = np.identity(4, float)

    print('Reference bone', ref_bone)
    for i in range(1, len(mask_names)):
        # print('i',i)
        target_bone = mask_names[i]
        print('target_bone', target_bone)

        if init_tibia and 'Patella' in target_bone:

            for j in range(1, im_no):
                transformPath1 = f'{path_reg}/Output/{ref_bone}/{(j):02d}/TransformParameters.0.txt'
                # print(transformPath1)\n",
                subjTx_ref[j] = PathToTx(transformPath1, sequence)  # retrieve uncombined transformation matrices

                transformPath_init = f'{path_reg}/Output/{mask_names[2]}/{(j):02d}/TransformParameters.0.txt'

                transformPath = f'{path_reg}/Output/{target_bone}/{(j):02d}/TransformParameters.0.txt'
                # print(transformPath)
                subjTx_target[j] = PathToTx(transformPath, sequence)  # retrieve uncombined transformation matrices
                init_Tx[j] = PathToTx(transformPath_init, sequence)
                # Tx1 = Compose4x4matrix(a1, b1, c1)
                combinedMatrix_pat[j] = subjTx_target[j].dot(combinedMatrix_pat[j - 1])
                combinedMatrix_tib[j] = init_Tx[j].dot(combinedMatrix_tib[j - 1])

            for up in range(fixed_pos[pos_selector] + 1, im_no, 1):
                # print(\"UP\",up)
                #             print(\"fixed pos\",fixed_pos[pos_selector] + 1)

                # subjTx_comb[up] = subjTx[up] @ subjTx_comb[up - 1]
                subjTx_comb_ref[up] = subjTx_ref[up].dot(subjTx_comb_ref[up - 1])
                # subjTx_comb_target[up] = subjTx_target[up].dot(subjTx_comb_target[up - 1])
                subjTx_comb_target[up] = (subjTx_target[up].dot(combinedMatrix_tib[up]))

                relativeMatrix = compute_relative_motion(subjTx_comb_ref[up], subjTx_comb_target[up])

                relativeMatrix_ref = compute_relative_motion(subjTx_comb_ref[up], subjTx_comb_ref[up])

                translation_combined = t3d.translation_from_matrix(relativeMatrix)
                angles_combined = t3d.euler_from_matrix(relativeMatrix, axes='s' + sequence)

                # Create a Rotation object from the rotation part of the transformation matrix
                rotation = R.from_matrix(relativeMatrix[:3, :3])

                # Get the Euler angles using a specific order
                euler_angles = rotation.as_euler(sequence, degrees=True)
                # print('euler_angles',euler_angles)

                Tx_relative[up] = relativeMatrix
                Tx_relative_ref[up] = relativeMatrix_ref
                np.savetxt(f'{path_reg}/Output/Relative_Transform/{target_bone}_rel_{ref_bone}_{(up):02d}_tx.txt',
                           relativeMatrix)
                np.savetxt(f'{path_reg}/Output/Relative_Transform_ref/{ref_bone}_rel_{ref_bone}_{(up):02d}_tx.txt',
                           relativeMatrix_ref)

                cardan_angles[up][2] = angles_combined[2] * 57.295779513
                cardan_angles[up][1] = angles_combined[1] * 57.295779513
                cardan_angles[up][0] = angles_combined[0] * 57.295779513

                euler_angles_res[up][2] = euler_angles[2]
                euler_angles_res[up][1] = euler_angles[1]
                euler_angles_res[up][0] = euler_angles[0]

            for down in range(fixed_pos[pos_selector] - 1, 0, -1):
                # print(\"DOWN\", down)
                # print(\"fixed pos\",fixed_pos[pos_selector] - 1)
                # subjTx_comb[down] = subjTx[down] @ subjTx_comb[down + 1]
                subjTx_comb_ref[down] = subjTx_ref[down].dot(subjTx_comb_ref[down + 1])
                # subjTx_comb_target[down] = subjTx_target[down].dot(subjTx_comb_target[down + 1])

                subjTx_comb_target[down] = (subjTx_target[down].dot(combinedMatrix_tib[down]))
                # print('subjTx_comb_down',down,subjTx_comb[down])

                relativeMatrix = compute_relative_motion(subjTx_comb_ref[down], subjTx_comb_target[down])
                relativeMatrix_ref = compute_relative_motion(subjTx_comb_ref[down], subjTx_comb_ref[down])

                # Create a Rotation object from the rotation part of the transformation matrix
                rotation = R.from_matrix(relativeMatrix[:3, :3])

                # Get the Euler angles using a specific order
                euler_angles = rotation.as_euler('zxy', degrees=True)

                # print('euler_angles',euler_angles)

                Tx_relative[down] = relativeMatrix
                Tx_relative_ref[down] = relativeMatrix_ref

                np.savetxt(f'{path_reg}/Output/Relative_Transform/{target_bone}_rel_{ref_bone}_{(down):02d}_tx.txt',
                           relativeMatrix)

                np.savetxt(f'{path_reg}/Output/Relative_Transform_ref/{ref_bone}_rel_{ref_bone}_{(down):02d}_tx.txt',
                           relativeMatrix_ref)

                # Sequential_Tx_relative.append(Tx_relative[down])\n",

                # print('euler',euler_angles)

                translation_combined = t3d.translation_from_matrix(relativeMatrix)
                angles_combined = t3d.euler_from_matrix(relativeMatrix, axes='sxyz')
                # print('angles_combined', angles_combined[0] * 57.295779513, angles_combined[1] * 57.295779513,
                #       angles_combined[2] * 57.295779513)
                #            print('translation_combined',translation_combined)
                cardan_angles[down][2] = angles_combined[2] * 57.295779513
                cardan_angles[down][1] = angles_combined[1] * 57.295779513
                cardan_angles[down][0] = angles_combined[0] * 57.295779513

                euler_angles_res[down][2] = euler_angles[2]
                euler_angles_res[down][1] = euler_angles[1]
                euler_angles_res[down][0] = euler_angles[0]

            # save cardan angles to txt and csv files
            txtfilename = f'{path_reg}/Output/Cardan_angles_{target_bone}_rel_{ref_bone}.txt'
            csvfilename = f'{path_reg}/Output/Cardan_angles_{target_bone}_rel_{ref_bone}.csv'
            excelfilename = f'{path_reg}/Output/Cardan_angles_{target_bone}_rel_{ref_bone}.xlsx'

            time = np.arange(0, im_no - 1)

            save_Cardan_angles(cardan_angles, txtfilename, csvfilename, time)
            read_file = pd.read_csv(csvfilename)
            read_file.to_excel(excelfilename, index=None, header=True)

        else:
            for j in range(1, im_no):
                transformPath1 = f'{path_reg}/Output/{ref_bone}/{(j):02d}/TransformParameters.0.txt'
                # print(transformPath1)\n",
                subjTx_ref[j] = PathToTx(transformPath1, sequence)  # retrieve uncombined transformation matrices
                # Tx1 = PathToTx(transformPath1)
                # Tx1 = Compose4x4matrix(a1, b1, c1)
                # combinedMatrix = Tx1.dot(combinedMatrix)

                transformPath = f'{path_reg}/Output/{target_bone}/{(j):02d}/TransformParameters.0.txt'
                # print(transformPath)
                subjTx_target[j] = PathToTx(transformPath, sequence)  # retrieve uncombined transformation matrices
                # Tx1 = PathToTx(transformPath)\n",
                # Tx1 = Compose4x4matrix(a1, b1, c1)\n",
                # combinedMatrix1 = Tx1.dot(combinedMatrix1)\n",
            #             subjTx_comb_ref[fixed_pos[pos_selector]] = subjTx_ref[fixed_pos[pos_selector]]
            #             subjTx_comb_target[fixed_pos[pos_selector]] = subjTx_target[fixed_pos[pos_selector]]

            #             relativeMatrix=compute_relative_motion(subjTx_comb_ref[fixed_pos[pos_selector]],subjTx_comb_target[fixed_pos[pos_selector]])\n",

            #             Tx_relative[fixed_pos[pos_selector]]=relativeMatrix\n",

            for up in range(fixed_pos[pos_selector] + 1, im_no, 1):
                # print(\"UP\",up)\n",
                #             print(\"fixed pos\",fixed_pos[pos_selector] + 1)\n",

                # subjTx_comb[up] = subjTx[up] @ subjTx_comb[up - 1]\n",
                subjTx_comb_ref[up] = subjTx_ref[up].dot(subjTx_comb_ref[up - 1])
                subjTx_comb_target[up] = subjTx_target[up].dot(subjTx_comb_target[up - 1])
                # combinedMatrix = Tx1.dot(combinedMatrix)
                # print('subjTx_comb_up',up,subjTx_comb[up])
                #             print('i',i)\n",
                relativeMatrix = compute_relative_motion(subjTx_comb_ref[up], subjTx_comb_target[up])

                relativeMatrix_ref = compute_relative_motion(subjTx_comb_ref[up], subjTx_comb_ref[up])

                translation_combined = t3d.translation_from_matrix(relativeMatrix)

                angles_combined = t3d.euler_from_matrix(relativeMatrix, axes='s' + sequence)

                # Create a Rotation object from the rotation part of the transformation matrix\n",
                rotation = R.from_matrix(relativeMatrix[:3, :3])

                # Get the Euler angles using a specific order\n",
                euler_angles = rotation.as_euler(sequence, degrees=True)
                # print('euler_angles',euler_angles)\n",

                Tx_relative[up] = relativeMatrix
                Tx_relative_ref[up] = relativeMatrix_ref
                np.savetxt(f'{path_reg}/Output/Relative_Transform/{(up):02d}_tx.txt', relativeMatrix)
                np.savetxt(f'{path_reg}/Output/Relative_Transform_ref/{(up):02d}_tx.txt', relativeMatrix_ref)

                cardan_angles[up][2] = angles_combined[2] * 57.295779513
                cardan_angles[up][1] = angles_combined[1] * 57.295779513
                cardan_angles[up][0] = angles_combined[0] * 57.295779513

                euler_angles_res[up][2] = euler_angles[2]
                euler_angles_res[up][1] = euler_angles[1]
                euler_angles_res[up][0] = euler_angles[0]

            for down in range(fixed_pos[pos_selector] - 1, 0, -1):
                # print(\"DOWN\", down)\n",
                # print(\"fixed pos\",fixed_pos[pos_selector] - 1)\n",
                # subjTx_comb[down] = subjTx[down] @ subjTx_comb[down + 1]\n",
                subjTx_comb_ref[down] = subjTx_ref[down].dot(subjTx_comb_ref[down + 1])
                subjTx_comb_target[down] = subjTx_target[down].dot(subjTx_comb_target[down + 1])
                # print('subjTx_comb_down',down,subjTx_comb[down])

                relativeMatrix = compute_relative_motion(subjTx_comb_ref[down], subjTx_comb_target[down])
                relativeMatrix_ref = compute_relative_motion(subjTx_comb_ref[down], subjTx_comb_ref[down])

                # Create a Rotation object from the rotation part of the transformation matrix\n",
                rotation = R.from_matrix(relativeMatrix[:3, :3])

                # Get the Euler angles using a specific order\n",
                euler_angles = rotation.as_euler('zxy', degrees=True)

                # print('euler_angles',euler_angles)\n",

                Tx_relative[down] = relativeMatrix
                Tx_relative_ref[down] = relativeMatrix_ref

                np.savetxt(f'{path_reg}/Output/Relative_Transform/{(down):02d}_tx.txt', relativeMatrix)

                np.savetxt(f'{path_reg}/Output/Relative_Transform_ref/{(down):02d}_tx.txt', relativeMatrix_ref)

                # Sequential_Tx_relative.append(Tx_relative[down])\n",

                # print('euler',euler_angles)\n",

                translation_combined = t3d.translation_from_matrix(relativeMatrix)
                angles_combined = t3d.euler_from_matrix(relativeMatrix, axes='sxyz')
                # print('angles_combined', angles_combined[0] * 57.295779513, angles_combined[1] * 57.295779513
                #       angles_combined[2] * 57.295779513)
                #            print('translation_combined',translation_combined)
                cardan_angles[down][2] = angles_combined[2] * 57.295779513
                cardan_angles[down][1] = angles_combined[1] * 57.295779513
                cardan_angles[down][0] = angles_combined[0] * 57.295779513

                euler_angles_res[down][2] = euler_angles[2]
                euler_angles_res[down][1] = euler_angles[1]
                euler_angles_res[down][0] = euler_angles[0]

            # save cardan angles to txt and csv files
            txtfilename = f'{path_reg}/Output/Cardan_angles_{target_bone}_rel_{ref_bone}.txt'
            csvfilename = f'{path_reg}/Output/Cardan_angles_{target_bone}_rel_{ref_bone}.csv'
            excelfilename = f'{path_reg}/Output/Cardan_angles_{target_bone}_rel_{ref_bone}.xlsx'

            time = np.arange(0, im_no - 1)

            save_Cardan_angles(cardan_angles, txtfilename, csvfilename, time)
            read_file = pd.read_csv(csvfilename)
            read_file.to_excel(excelfilename, index=None, header=True)

        # print('euler',euler_angles_res)
        # print('cardan_angles',cardan_angles)
        # TX_matrix_file=f'{path_reg}/Output/TX_{target_bone}_rel_{ref_bone}.txt'
        # np.savetxt(TX_matrix_file,Tx_relative)

        pos_selector = pos_selector + 1
        return Tx_relative, array_of_matrices


# ANKLE
def atlas_labels_dicts_ankle(atlas_folder, side):
    '''atlas_folder = Folder containing the atlases
       side         = left or right
    '''

    atlas_dir = sorted(glob.glob(atlas_folder + '/*/' + side + '/*_rs.mhd'))

    atlas_labels_dir = sorted(glob.glob(atlas_folder + '/*/' + side + '/*GT.mhd'))

    landmarks_dir = sorted(glob.glob(atlas_folder + '/*/' + side + '/*tfx.txt'))

    atlas_files = [{'image': atlas, 'label': label, 'landmarks': landmark}
                   for atlas, label, landmark in zip(atlas_dir, atlas_labels_dir, landmarks_dir)]

    #     val_T1_dir=sorted(glob.glob(traindir+'/imagesTs/*'))

    #     val_labels_dir=sorted(glob.glob(traindir+'/labelsTs/*'))

    #     val_files=[{'image': [T1], 'label': label}
    #       for T1,label in zip(val_T1_dir,val_labels_dir)]

    return atlas_files


def atlas_labels_dicts_Knee(atlas_folder, side):
    '''atlas_folder = Folder containing the atlases
       side         = left or right
    '''

    atlas_dir = sorted(glob.glob(atlas_folder + '/*/' + side + '/*0.mhd'))
    print('atlas_dir',atlas_folder)


    atlas_labels_dir = sorted(glob.glob(atlas_folder + '/*/' + side + '/*GT.mhd'))

    landmarks_dir = sorted(glob.glob(atlas_folder + '/*/' + side + '/points_LB_tfx.txt'))



    # landmarks_dir=sorted(glob.glob(atlas_folder+'/*/'+side+'/landmarks_tfx_mean_readers.txt'))

    atlas_files = [{'image': atlas, 'label': label, 'landmarks': landmark}
                   for atlas, label, landmark in zip(atlas_dir, atlas_labels_dir, landmarks_dir)]

    #     val_T1_dir=sorted(glob.glob(traindir+'/imagesTs/*'))

    #     val_labels_dir=sorted(glob.glob(traindir+'/labelsTs/*'))

    #     val_files=[{'image': [T1], 'label': label}
    #       for T1,label in zip(val_T1_dir,val_labels_dir)]

    return atlas_files


def Refine_segmentation_thumb(inputdatapath, outputdata, radius, Thresh1, Thresh2, Thresh3):
    kernel = (radius, radius, radius)
    inputdata = sitk.ReadImage(inputdatapath)

    # original segmented images
    MC1_og = sitk.BinaryThreshold(inputdata, Thresh1, Thresh1)
    TZ_og = sitk.BinaryThreshold(inputdata, Thresh2, Thresh2)
    MC2_og = sitk.BinaryThreshold(inputdata, Thresh3, Thresh3)

    MC1path_og = outputdata + '/MC1_final.mhd'
    TZpath_og = outputdata + '/TZ_final.mhd'
    MC2path_og = outputdata + '/MC2_final.mhd'

    # binarize images
    MC1 = sitk.BinaryThreshold(inputdata, Thresh1, Thresh1)
    TZ = sitk.BinaryThreshold(inputdata, Thresh2, Thresh2)
    MC2 = sitk.BinaryThreshold(inputdata, Thresh3, Thresh3)
    # Femur = sitk.BinaryThreshold(inputdata, Thresh3, Thresh3)

    # morphological image closing
    MC1 = sitk.BinaryMorphologicalClosing(MC1, kernel)
    TZ = sitk.BinaryMorphologicalClosing(TZ, kernel)
    MC2 = sitk.BinaryMorphologicalClosing(MC2, kernel)
    # Femur = sitk.BinaryMorphologicalClosing(Femur, kernel)

    # morphological dilation
    MC1 = sitk.BinaryDilate(MC1, kernel)
    TZ = sitk.BinaryDilate(TZ, kernel)
    MC2 = sitk.BinaryDilate(MC2, kernel)
    # Femur = sitk.BinaryDilate(Femur, kernel)

    MC1path = outputdata + '/MC1_final_r' + str(radius) + '.mhd'
    TZpath = outputdata + '/TZ_final_r' + str(radius) + '.mhd'
    MC2path = outputdata + '/MC2_final_r' + str(radius) + '.mhd'
    # femurpath = outputdata + '/Femur_final_r' + str(radius) + '.mhd'

    sitk.WriteImage(MC1, MC1path)
    sitk.WriteImage(TZ, TZpath)
    sitk.WriteImage(MC2, MC2path)


def Refine_segmentation_knee(inputdatapath, outputdata, radius):
    kernel = (radius, radius, radius)
    inputdata = sitk.ReadImage(inputdatapath)

    # original segmented images
    Tibia_og = sitk.BinaryThreshold(inputdata, 1, 1)
    femur_og = sitk.BinaryThreshold(inputdata, 2, 2)
    patella_og = sitk.BinaryThreshold(inputdata, 3, 3)

    Tibiapath_og = outputdata + '/Tibia_final.mhd'
    femurpath_og = outputdata + '/Femur_final.mhd'
    patellapath_og = outputdata + '/Patella_final.mhd'

    # binarize images
    Tibia = sitk.BinaryThreshold(inputdata, 1, 1)
    Femur = sitk.BinaryThreshold(inputdata, 2, 2)
    Patella = sitk.BinaryThreshold(inputdata, 3, 3)
    # Femur = sitk.BinaryThreshold(inputdata, Thresh3, Thresh3)

    # morphological image closing
    Tibia = sitk.BinaryMorphologicalClosing(Tibia, kernel)
    Femur = sitk.BinaryMorphologicalClosing(Femur, kernel)
    Patella = sitk.BinaryMorphologicalClosing(Patella, kernel)
    # Femur = sitk.BinaryMorphologicalClosing(Femur, kernel)

    # morphological dilation
    Tibia = sitk.BinaryDilate(Tibia, kernel)
    Femur = sitk.BinaryDilate(Femur, kernel)
    Patella = sitk.BinaryDilate(Patella, kernel)
    # Femur = sitk.BinaryDilate(Femur, kernel)

    Tibiapath = outputdata + '/Tibia_final_r' + str(radius) + '.mhd'
    Femurpath = outputdata + '/Femur_final_r' + str(radius) + '.mhd'
    Patellapath = outputdata + '/Patella_final_r' + str(radius) + '.mhd'
    # femurpath = outputdata + '/Femur_final_r' + str(radius) + '.mhd'

    sitk.WriteImage(Tibia, Tibiapath)
    sitk.WriteImage(Femur, Femurpath)
    sitk.WriteImage(Patella, Patellapath)

    sitk.WriteImage(Tibia_og, Tibiapath_og)
    sitk.WriteImage(femur_og, femurpath_og)
    sitk.WriteImage(patella_og, patellapath_og)
    # sitk.WriteImage(Femur, femurpath)


def Refine_segmentation_ankle(inputdatapath, outputdata, radius):
    kernel = (radius, radius, radius)
    inputdata = sitk.ReadImage(inputdatapath)

    # original segmented images
    Calcaneus_og = sitk.BinaryThreshold(inputdata, 1, 1)
    Talus_og = sitk.BinaryThreshold(inputdata, 2, 2)
    Tibia_og = sitk.BinaryThreshold(inputdata, 3, 3)

    Calcaneuspath_og = outputdata + '/Calcaneus_final.mhd'
    Taluspath_og = outputdata + '/Talus_final.mhd'
    Tibiapath_og = outputdata + '/Tibia_final.mhd'

    # binarize images
    Calcaneus = sitk.BinaryThreshold(inputdata, 1, 1)
    Talus = sitk.BinaryThreshold(inputdata, 2, 2)
    Tibia = sitk.BinaryThreshold(inputdata, 3, 3)
    # Femur = sitk.BinaryThreshold(inputdata, Thresh3, Thresh3)

    # morphological image closing
    Calcaneus = sitk.BinaryMorphologicalClosing(Calcaneus, kernel)
    Talus = sitk.BinaryMorphologicalClosing(Talus, kernel)
    Tibia = sitk.BinaryMorphologicalClosing(Tibia, kernel)
    # Femur = sitk.BinaryMorphologicalClosing(Femur, kernel)

    # morphological dilation
    Calcaneus = sitk.BinaryDilate(Calcaneus, kernel)
    Talus = sitk.BinaryDilate(Talus, kernel)
    Tibia = sitk.BinaryDilate(Tibia, kernel)
    # Femur = sitk.BinaryDilate(Femur, kernel)

    Calcaneuspath = outputdata + '/Calcaneus_final_r' + str(radius) + '.mhd'
    Taluspath = outputdata + '/Talus_final_r' + str(radius) + '.mhd'
    Tibiapath = outputdata + '/Tibia_final_r' + str(radius) + '.mhd'
    # femurpath = outputdata + '/Femur_final_r' + str(radius) + '.mhd'

    sitk.WriteImage(Calcaneus, Calcaneuspath)
    sitk.WriteImage(Talus, Taluspath)
    sitk.WriteImage(Tibia, Tibiapath)
    # sitk.WriteImage(Femur, femurpath)


# DRAW PATCH AROUND LANDMARK
def SIFT_Mask(PointsFile, bounds, Image, indices):
    """
    PointsFile: file containing sift points
    bounds: size of the mask to be drawn arround the sift point
    Image: fixed or moving image
    Indices: index of the sift point in PointsFile
    """
    import copy

    # make a numpy array copy of the image for numpy manipulations
    image_copy = sitk.Image(Image.GetSize(), sitk.sitkUInt16)
    image_copy.CopyInformation(Image)
    npcopy = sitk.GetArrayFromImage(image_copy)
    # print(fixed)

    # initialize
    point_max = [0, 0, 0]
    point_min = [0, 0, 0]

    point_max2 = [0, 0, 0]
    point_min2 = [0, 0, 0]

    # select one of the sift points and transform to Index
    # S=Image.TransformPhysicalPointToIndex(PointsFile[indices])

    # initialise an empty point to hold the sift cordinates
    pt = [0, 0, 0]

    # using physical cordinate
    pt[0] = PointsFile[indices][0]
    pt[1] = PointsFile[indices][1]
    pt[2] = PointsFile[indices][2]

    # debug
    #     print("point",pt)
    # print(bounds)

    # Manipulate( add the size of the rectangle) the points
    point_max[0] = (pt[0] + bounds)
    point_max[1] = pt[1] + bounds
    point_max[2] = (pt[2] + bounds)

    point_min[0] = pt[0] - bounds
    point_min[1] = pt[1] - bounds
    point_min[2] = (pt[2] - bounds)

    # convert physical point to index
    point_max = Image.TransformPhysicalPointToIndex(point_max)
    point_min = Image.TransformPhysicalPointToIndex(point_min)

    #     print("maxpts:",point_max)
    #     print("minpts:",point_min)

    # handle borders
    point_max2[0] = min(point_max[0], npcopy.shape[2] - 1)
    point_max2[1] = min(point_max[1], npcopy.shape[1] - 1)
    point_max2[2] = min(point_max[2], npcopy.shape[0] - 1)

    # debug
    # print("pmax:",point_max)

    point_min2[0] = max(point_min[0], 0)
    point_min2[1] = max(point_min[1], 0)
    point_min2[2] = max(point_min[2], 0)

    #     print("maxpts after border:",point_max2)
    #     print("minpts after border:",point_min2)

    ##create the mask
    im = np.zeros_like(npcopy)  #
    for x in range(point_min2[0], point_max2[0]):
        for y in range(point_min2[1], point_max2[1]):
            for z in range(point_min2[2], point_max2[2]):
                if ((x >= point_min2[0] and x <= point_max2[0]) and (y >= point_min2[1] and y <= point_max2[1]) and (
                        z >= point_min2[2] and z <= point_max2[2])):
                    im[z, y, x] = 255
                    # print(im[z,y,x])
    SIFT_mask = sitk.GetImageFromArray(im)

    # ensure mask and input image are in the same space by copying the image info and convert to uchar
    SIFT_mask.CopyInformation(image_copy)
    SIFT_mask = sitk.Cast(SIFT_mask, sitk.sitkUInt8)

    # optionally save the mask
    # sitk.WriteImage(SIFT_mask, "/Users/bkeelson/Conferences/MICCAI/Result/SIFT_Mask.mhd")

    # SIFT_mask multiplied by original image
    inImage = sitk.Mask(Image, SIFT_mask)
    # return the mask
    # image=sitk.Mask(Image,SIFT_Mask)

    # crop image
    Cpmax = [npcopy.shape[2] - point_max2[0], npcopy.shape[1] - point_max2[1],
             npcopy.shape[0] - point_max2[2]]  # determin max crop size(considering numpy notation)
    croppedImage = sitk.Crop(Image, point_min2, Cpmax)

    # return cropped image
    return croppedImage, pt


##idea .. add a "if debug" mode to save intermediate results else you delete all intermediates at the end


total_start = time.time()


def NCC_point_fusion_v2(fixedimage, movingimages, outputfolder, mask_folder, nr_atlas, atlas_list, vv_template_path,
                        kernel):
    meanlist = []
    #
    vv_template = np.genfromtxt(vv_template_path, skip_header=1, dtype='float', delimiter=' ')

    atlas_ncc = []
    mydata = np.empty([len(vv_template), len(atlas_list)])

    j = 0
    print("ESTIMATING IDEAL POINTS ")
    # for atlas in atlas_list:
    for i, atlas in enumerate(atlas_list):
        movingimage = movingimages[i]
        Path_landmark = outputfolder + '/' + atlas + '/landmarks_tfx_' + atlas + '.txt'
        for i in range(0, len(vv_template)):
            #         print(Path_landmark)
            try:
                landmarks = np.genfromtxt(Path_landmark, skip_header=2, filling_values=np.NaN)
            except:
                continue
            # print("POINT", i)

        for i in range(0, len(vv_template)):
            try:


                fixedMask, _ = SIFT_Mask(landmarks, kernel, fixedimage, i)
                movingMask, _ = SIFT_Mask(landmarks, kernel, movingimage, i)
                cast_Moving = sitk.Cast(movingMask, fixedMask.GetPixelID())
                NMC = sitk.FFTNormalizedCorrelation(fixedMask, cast_Moving)
                stats = sitk.StatisticsImageFilter()
                # ensure images occupy the same physical space

                stats.Execute(NMC)

                # USE variance
                varianceNCC = stats.GetVariance()

                mydata[i][j] = varianceNCC
                atlas_ncc.append(varianceNCC)

            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)

        #             except:
        #                 continue
        j = j + 1
    data_num = np.array(mydata)

    import pandas as pd

    # creating a DataFrame object
    df = pd.DataFrame(data_num,
                      index=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12',
                             'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24'],
                      columns=atlas_list)
    df.head()
    # get indices with max NCC
    maxvalueIndexLabel = df.idxmax(axis=1)
    trial = df.columns.values[np.argsort(-df.values, axis=1)[:, :nr_atlas]]


    i = 0
    for mlist in trial:
        print("The ", nr_atlas, " best atlases for landmark: ", i, " are ", mlist)
        myptsFinal = []
        for m_atlas in range(0, nr_atlas):

            Path_landmark = outputfolder + '/' + mlist[m_atlas] + '/landmarks_tfx_' + mlist[m_atlas] + '.txt'
            try:
                landmarks = np.genfromtxt(Path_landmark, skip_header=2, filling_values=np.NaN)
            except:
                continue

            myptsFinal.append(landmarks[i])

        mynumpy = np.array(myptsFinal)

        avgmynumpy = np.mean(mynumpy, axis=0)
        meanlist.append(avgmynumpy)

        i = i + 1

    numpyMeanlist = np.array(meanlist)
    landmarks_vv = mask_folder + '/allmaxlandmarks_vv_NCC_%d_.txt' % kernel
    landmarks_tfx = mask_folder + '/allmaxlandmarks_tfx_NCC_%d_.txt' % kernel

    create_vv_pts_from_numpypts(vv_template_path, numpyMeanlist, landmarks_vv, landmarks_tfx)

    landmarkfolder =  os.path.dirname(outputfolder)+ '/points'
    # Create landmark output directory if it doesn't exist
    if not os.path.exists(landmarkfolder):
        os.makedirs(landmarkfolder)

    shutil.copy2(landmarks_vv, landmarkfolder + '/points.txt')


def VV_pts_to_tfx(path2pts, outputpath):
    points = np.genfromtxt(path2pts, skip_header=1, filling_values=np.NaN)

    points_transf = points[:, 1:4]
    points_count = str(np.size(points, 0))

    np.savetxt(outputpath, points_transf, delimiter=' ')

    file = []
    with open(outputpath,
              'r') as read_the_whole_thing_first:
        for line in read_the_whole_thing_first:
            file.append(line)
    file = [points_count + '\n'] + file
    file = ["point\n"] + file
    with open(outputpath, 'r+') as f:
        for line in file:
            f.writelines(line)
    with open(outputpath, 'r') as f2:
        a = f2.read()



# takes the manually selected points on the fixed image and converts them into transformix style points
def Create_Transformix_pts(path2pts, outputpath):
    points = np.genfromtxt(path2pts, skip_header=1, filling_values=np.NaN)

    points_transf = points[:, 1:4]
    points_count = str(np.size(points, 0))

    np.savetxt(outputpath, points_transf, delimiter=' ')

    file = []
    with open(outputpath,
              'r') as read_the_whole_thing_first:
        for line in read_the_whole_thing_first:
            file.append(line)
    file = [points_count + '\n'] + file
    file = ["point\n"] + file
    with open(outputpath, 'r+') as f:
        for line in file:
            f.writelines(line)
    with open(outputpath, 'r') as f2:
        a = f2.read()


###making vv style points from output pts of transformix#######
def create_vv_pts_from_outputtfx(vv_template_path, transformix_pts_path, outputpath, tfx_pts_file):
    """vv_template_path is the path to an original vv points file
    transformix_pts is the output file from transformix after transforming some pts
    outputpath is the resulting vv style points converted from transformix pts
    """

    vv_template = np.genfromtxt(vv_template_path, skip_header=1, dtype='float', delimiter=' ')
    transformix_pts = np.genfromtxt(transformix_pts_path, dtype='float', delimiter=' ')
    copy_template = vv_template.copy()
    extract_tfx = transformix_pts[:, 25:28]

    for i in range(0, len(copy_template)):  # iterate over number of points(rows)
        for j in range(1, 4):  # iterate over columns
            # assign transformix obtained pts to the vv template
            copy_template[i][j] = extract_tfx[i][j - 1]

    np.savetxt(outputpath, copy_template, delimiter=' ')
    file = []
    with open(outputpath,
              'r') as read_the_whole_thing_first:
        for line in read_the_whole_thing_first:
            file.append(line)
    file = ["LANDMARKS1\n"] + file
    with open(outputpath, 'r+') as f:
        for line in file:
            f.writelines(line)
    with open(outputpath, 'r') as f2:
        a = f2.read()

    # create points to be used by transformix
    raw_tfx_pts = VV_pts_to_tfx(outputpath, tfx_pts_file)

    return raw_tfx_pts


def create_vv_pts_from_numpypts(vv_template_path, numpy_pts, outputpath, tfx_pts_file):
    """vv_template_path is the path to an original vv points file
    transformix_pts is the output file from transformix after transforming some pts
    outputpath is the resulting vv style points converted from transformix pts
    """

    vv_template = np.genfromtxt(vv_template_path, skip_header=1, dtype='float', delimiter=' ')
    #     transformix_pts = np.genfromtxt(transformix_pts_path, dtype='float', delimiter=' ')
    copy_template = vv_template.copy()
    #     extract_tfx=transformix_pts[:, 25:28]

    for i in range(0, len(copy_template)):  # iterate over number of points(rows)
        for j in range(1, 4):  # iterate over columns
            # assign transformix obtained pts to the vv template
            copy_template[i][j] = numpy_pts[i][j - 1]

    np.savetxt(outputpath, copy_template, delimiter=' ')
    file = []
    with open(outputpath,
              'r') as read_the_whole_thing_first:
        for line in read_the_whole_thing_first:
            file.append(line)
    file = ["LANDMARKS1\n"] + file
    with open(outputpath, 'r+') as f:
        for line in file:
            f.writelines(line)
    with open(outputpath, 'r') as f2:
        a = f2.read()

    # create points to be used by transformix
    raw_tfx_pts = VV_pts_to_tfx(outputpath, tfx_pts_file)

    return raw_tfx_pts


# Define paths to fixed and moving images

# fixed_image_path = '/Volumes/Luca_T5/4D_MSK/Dynamic_data/Ankle/Brace/s01p14.mhd'
def run_MAS_ELASTIX (general_path, fixed_image_path, joint, side, outputfolder,radius=3):
    total_start = time.time()
    print('Mutli-Atlas segmentation of ', fixed_image_path, 'has started')
    # Define path to parameter file

    rigid = general_path + '/Scripts/Param_Files/Euler_Without_Mask_MSD.txt'
    affine = general_path + '/Scripts/Param_Files/Affine_Without_Mask_MSD.txt'
    bspline = general_path + '/Scripts/Param_Files/BSpline_Without_Mask_MSD.txt'
    bspline_inverse = general_path + '/Scripts/Param_Files/BSpline_Without_Mask_INVERSE.txt'

    # Define path to label fusion parameter file
    label_fusion_parameter_file_path = 'label_fusion.txt'

    # general_path='/Volumes/Luca_T5/4D_MSK'
    # atlas_folder=f'{general_path}/Atlas'
    atlas_folder = f'{general_path}/Atlas/' + joint + '/Source/'
    # side='right'

    # Define output directory
    #output_directory = f'{general_path}/Atlas/{subject}'
    output_directory = outputfolder

    # Create output directory if it doesn't exist

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print('created directory', output_directory)

    # Load fixed image
    fixed_image = sitk.ReadImage(fixed_image_path)

    ## call function that creates dictionaries from atlas folder
    atlas_data_paths = atlas_labels_dicts_Knee(atlas_folder, side)

    print('atlas_data_paths ', atlas_data_paths )

    # Load atlas images, labels and landmarks
    atlas_images = []
    label_images = []
    anat_landmarks = []
    atlas_list = []

    # Set up registration
    transformixImageFilter = sitk.TransformixImageFilter()
    registration = sitk.ElastixImageFilter()
    registration.SetFixedImage(fixed_image)
    registration.SetParameterMap(registration.ReadParameterFile(rigid))
    registration.AddParameterMap(registration.ReadParameterFile(affine))
    registration.AddParameterMap(registration.ReadParameterFile(bspline))

    transformixImageFilter.LogToConsoleOff()
    #     registration.LogToConsoleOn()

    # Set up inverse registration
    registration_inverse = sitk.ElastixImageFilter()
    #     registration_inverse.SetFixedImage(fixed_image)
    #     registration_inverse.SetParameterMap(registration.ReadParameterFile(bspline_inverse))
    #     registration_inverse.LogToConsoleOff()

    # Register each atlas to the fixed image
    registered_atlas_images = []
    transformed_labels = []
    transformed_landmarks = []

    for files in (atlas_data_paths):
        print('Running atlas : ', files['image'], ' Fixed image : ', fixed_image_path)

        folder_name = files['image'].split('/')[-3]
        atlas_list.append(folder_name)

        output_atlas_folder = output_directory + '/' + folder_name
        if not os.path.exists(output_atlas_folder):
            os.makedirs(output_atlas_folder)
            print('created output_atlas_folder', output_atlas_folder)

        output_atlas_folder_inverse = os.path.join(output_atlas_folder, 'inverse')

        if not os.path.exists(output_atlas_folder_inverse):
            os.makedirs(output_atlas_folder_inverse)
            print('created output_atlas_folder_inverse', output_atlas_folder_inverse)

        atlas_image = sitk.ReadImage(files['image'])
        label_image = sitk.ReadImage(files['label'])
        landmarks = files['landmarks']
        atlas_images.append(atlas_image)
        label_images.append(label_image)
        anat_landmarks.append(landmarks)

        registration.LogToConsoleOff()

        registration.SetMovingImage(atlas_image)
        registration.SetOutputDirectory(output_atlas_folder)

        try:
            registration.Execute()
        except:
            continue
        registered_atlas_image = registration.GetResultImage()
        registered_atlas_images.append(registered_atlas_image)




        parameterMap = registration.GetTransformParameterMap()

        transformParameterFile = os.path.join(output_atlas_folder, 'TransformParameters.2.txt')


        parameterMap[2]['FinalBSplineInterpolationOrder'] = ['0']
        parameterMap[2]['DefaultPixelValue'] = ['0']
        parameterMap[2]['ResultImagePixelType'] = ['unsigned char']

        transformixImageFilter.SetMovingImage(label_image)
        transformixImageFilter.SetTransformParameterMap(parameterMap)
        transformixImageFilter.Execute()
        transformed_label = transformixImageFilter.GetResultImage()

        # cast to unchar
        mask = sitk.Cast(transformed_label, sitk.sitkUInt8)
        transformed_labels.append(mask)
        transformed_mask = output_atlas_folder + '/transformed_mask.nii'

        # if debug: save result for each transformed label
        sitk.WriteImage(mask, transformed_mask)

        print('output_atlas_folder_inverse', output_atlas_folder_inverse)
        subprocess.run(['elastix', '-f', fixed_image_path, '-m', fixed_image_path, '-p', parameter_file_path, '-out',
                        output_atlas_folder_inverse, '-t0', transformParameterFile])

        registration_inverse = sitk.ElastixImageFilter()
        inverse_T0 = registration_inverse.ReadParameterFile(output_atlas_folder_inverse + '/TransformParameters.0.txt')
        registration_inverse.SetParameterMap(inverse_T0)

        _inverse = registration_inverse.PrintParameterMap()
        # parameterMap_inverse = registration_inverse.GetTransformParameterMap()
        parameterMap_inverse = _inverse.GetParameterMap()
        parameterMap_inverse[0]['InitialTransformParametersFileName'] = ['NoInitialTransform']

        ## transform landmarks using the inverse
        transformixImageFilter.SetTransformParameterMap(parameterMap_inverse)
        transformixImageFilter.SetFixedPointSetFileName(landmarks)
        transformixImageFilter.SetOutputDirectory(output_atlas_folder)
        transformixImageFilter.Execute()

        ### extract transformed landmarks from outputpoints.txt
        transformix_pts_path = output_atlas_folder + '/outputpoints.txt'
        outputpath = output_atlas_folder + '/outputpoints_vv.txt'
        tfx_pts_file = output_atlas_folder + '/landmarks_tfx_' + folder_name + '.txt'
        # vv_template_path = '/home/pieter/Documents/4D_MSK/Atlas/Knee/Source/16/left/points.txt'
        # vv_template_path='/Users/bkeelson/4DMSK/vv_template_landmarks.txt'
        vv_template_path = sorted(glob.glob(atlas_folder + '/*/' + side + '/points_LB_all.txt'))[0]
        print('vv_template_path1', vv_template_path)

        print('creating output points vv')
        raw_tfx_pts = create_vv_pts_from_outputtfx(vv_template_path, transformix_pts_path, outputpath, tfx_pts_file)

#    fusionfilter = sitk.MultiLabelSTAPLEImageFilter()
#    fusionfilter.SetLabelForUndecidedPixels(0)
#
#    segmentation_image = fusionfilter.Execute(transformed_labels)

    mask_folder = outputfolder + '/Mask'
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    # segmentation_image=fusionfilter.GetResultImage()

    # reference_segmentation_STAPLE = filter.Execute(segmentations)

    # labelForUndecidedPixels=0

    # reference_STAPLE = sitk.MultiLabelSTAPLE(segmentations,
    #                                          labelForUndecidedPixels)
    # sitk.WriteImage(reference_STAPLE,'reference_STAPLE.mha')
    # final_segmentation_folder=

#     if not os.path.exists(outputfolder):
#         os.makedirs(outputfolder)
#
#     seg_output=outputfolder+'/segmentation.nii'
#     # Save segmentation image
#     sitk.WriteImage(segmentation_image, seg_output)
#     print("Splitting segmentations into individual bones")
#     if not os.path.exists(outputfolder):
#             os.makedirs(outputfolder)
#
#     if joint=='Ankle':
#         Refine_segmentation_ankle(seg_output, outputfolder, radius)
#
#     if joint=='Knee':
#         Refine_segmentation_knee(seg_output, outputfolder, radius)
#
#     if joint=='Wrist':
#         Refine_segmentation_thumb(seg_output, outputfolder, radius)
#
#     else:
#         print('Unknown joint')

    duration=time.time() - total_start

    time_format = time.strftime("%H:%M:%S", time.gmtime(duration))
    print('Writing results to ',outputfolder)

    print('Multi-atlas segmentation of  completed in :' ,time_format)

    duration = time.time() - total_start

    time_format = time.strftime("%H:%M:%S", time.gmtime(duration))
    print('Writing results to ', mask_folder)

    ###fusion of landmarks####
    print('Running NCC_point_fusion_v2')
    movingimages = registered_atlas_images
    nr_atlas = 3
    kernel = 3

    NCC_point_fusion_v2(fixed_image, registered_atlas_images, outputfolder, mask_folder, nr_atlas, atlas_list,
                        vv_template_path, kernel)

    print('Multi-atlas segmentation of  completed in :', time_format)

##make a dictionary of atlas, label and landmarks.

def run_MAS_og(general_path, fixed_image_path, joint, side, outputfolder, radius=3):
    total_start = time.time()
    print('Mutli-Atlas segmentation of ', fixed_image_path, 'has started')
    # Define path to parameter file

    rigid = general_path + '/Scripts/Param_Files/Euler_Without_Mask_MSD.txt'
    affine = general_path + '/Scripts/Param_Files/Affine_Without_Mask_MSD.txt'
    bspline = general_path + '/Scripts/Param_Files/BSpline_Without_Mask_MSD.txt'
    bspline_inverse = general_path + '/Scripts/Param_Files/BSpline_Without_Mask_INVERSE.txt'

    # Define path to label fusion parameter file
    label_fusion_parameter_file_path = 'label_fusion.txt'

    # general_path='/Volumes/Luca_T5/4D_MSK'
    # atlas_folder=f'{general_path}/Atlas'
    atlas_folder = f'{general_path}/Atlas/' + joint + '/Source/'

    print(atlas_folder)
    # side='right'

    # Define output directory
    # output_directory = f'{general_path}/Atlas/{subject}'
    output_directory = outputfolder

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load fixed image
    fixed_image = sitk.ReadImage(fixed_image_path)

    ## call function that creates dictionaries from atlas folder
    if joint == 'Ankle':
        atlas_data_paths = atlas_labels_dicts_ankle(atlas_folder, side)

    else:
        atlas_data_paths = atlas_labels_dicts_Knee(atlas_folder, side)

    # Load atlas images, labels and landmarks
    atlas_images = []
    label_images = []
    anat_landmarks = []

    # Set up registration
    transformixImageFilter = sitk.TransformixImageFilter()
    registration = sitk.ElastixImageFilter()
    registration.SetFixedImage(fixed_image)
    registration.SetParameterMap(registration.ReadParameterFile(rigid))
    registration.AddParameterMap(registration.ReadParameterFile(affine))
    registration.AddParameterMap(registration.ReadParameterFile(bspline))

    transformixImageFilter.LogToConsoleOff()
    registration.LogToConsoleOff()

    # Set up inverse registration
    registration_inverse = sitk.ElastixImageFilter()
    registration_inverse.SetFixedImage(fixed_image)
    registration_inverse.SetParameterMap(registration.ReadParameterFile(bspline_inverse))
    registration_inverse.LogToConsoleOff()

    # Register each atlas to the fixed image
    registered_atlas_images = []
    transformed_labels = []
    transformed_landmarks = []
    atlas_list = []
    outputfolder_list = []

    for files in tqdm(atlas_data_paths):
        print('Running atlas : ', files['image'], ' Fixed image : ', fixed_image_path)
        atlas_image = sitk.ReadImage(files['image'])
        label_image = sitk.ReadImage(files['label'])
        landmarks = files['landmarks']
        atlas_images.append(atlas_image)
        label_images.append(label_image)
        anat_landmarks.append(landmarks)
        # registration.LogToConsoleOn()

        registration.SetMovingImage(atlas_image)
        folder_name = files['image'].split('/')[8]

        atlas_list.append(folder_name)

        output_atlas_folder = output_directory + '/' + folder_name
        outputfolder_list.append(output_atlas_folder)
        if not os.path.exists(output_atlas_folder):
            os.makedirs(output_atlas_folder)

        registration.SetOutputDirectory(output_atlas_folder)

        try:
            registration.Execute()
        except:
            continue
        registered_atlas_image = registration.GetResultImage()
        registered_atlas_images.append(registered_atlas_image)

        parameterMap = registration.GetTransformParameterMap()
        parameterMap_landmarks = registration.GetTransformParameterMap()

        parameterMap[2]['FinalBSplineInterpolationOrder'] = ['0']
        parameterMap[2]['DefaultPixelValue'] = ['0']
        parameterMap[2]['ResultImagePixelType'] = ['unsigned char']

        transformixImageFilter.SetMovingImage(label_image)
        transformixImageFilter.SetTransformParameterMap(parameterMap)
        transformixImageFilter.Execute()
        transformed_label = transformixImageFilter.GetResultImage()

        # cast to unchar
        mask = sitk.Cast(transformed_label, sitk.sitkUInt8)
        transformed_labels.append(mask)
        transformed_mask = output_atlas_folder + '/transformed_mask.nii'

        # if debug: save result for each transformed label
        sitk.WriteImage(mask, transformed_mask)

        # inverse
        transformParameterFile = os.path.join(output_atlas_folder, 'TransformParameters.2.txt')
        print('InitialTransformParameterFileName', transformParameterFile)
        registration_inverse.SetMovingImage(fixed_image)
        registration_inverse.SetInitialTransformParameterFileName(transformParameterFile)

        output_atlas_folder_inverse = output_atlas_folder + '/inverse'
        if not os.path.exists(output_atlas_folder_inverse):
            os.makedirs(output_atlas_folder_inverse)

        registration_inverse.SetOutputDirectory(output_atlas_folder_inverse)
        registration_inverse.Execute()
        parameterMap_inverse = registration_inverse.GetTransformParameterMap()
        parameterMap_inverse[0]['InitialTransformParametersFileName'] = ['NoInitialTransform']

        ## transform landmarks using the inverse
        transformixImageFilter.SetTransformParameterMap(parameterMap_inverse)
        transformixImageFilter.SetFixedPointSetFileName(landmarks)
        transformixImageFilter.SetOutputDirectory(output_atlas_folder)
        transformixImageFilter.Execute()

        ### extract transformed landmarks from outputpoints.txt
        transformix_pts_path = output_atlas_folder + '/outputpoints.txt'
        outputpath = output_atlas_folder + '/outputpoints_vv.txt'
        tfx_pts_file = output_atlas_folder + '/landmarks_tfx_' + folder_name + '.txt'
        vv_template_path = '/home/pieter/Documents/4D_MSK/Atlas/Knee/Source/16/left/points.txt'

        print('creating output points vv')
        raw_tfx_pts = create_vv_pts_from_outputtfx(vv_template_path, transformix_pts_path, outputpath, tfx_pts_file)

    #     fusionfilter = sitk.MultiLabelSTAPLEImageFilter()
    #     fusionfilter.SetLabelForUndecidedPixels(0)

    #     segmentation_image = fusionfilter.Execute(transformed_labels)
    mask_folder = outputfolder + '/Mask'
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    #     seg_output = mask_folder + '/segmentation.nii'
    #     # Save segmentation image
    #     sitk.WriteImage(segmentation_image, seg_output)
    #     print("Splitting segmentations into individual bones")

    #     if joint == 'Ankle':
    #         Refine_segmentation_ankle(seg_output, mask_folder, radius)

    #     if joint == 'Knee':
    #         Refine_segmentation_knee(seg_output, mask_folder, radius)

    #     if joint == 'Wrist':
    #         Refine_segmentation_thumb(seg_output, mask_folder, radius)

    #     else:
    #         print('Unknown joint')

    duration = time.time() - total_start

    time_format = time.strftime("%H:%M:%S", time.gmtime(duration))
    print('Writing results to ', outputfolder)
    vv_template_path = '/home/pieter/Documents/4D_MSK/Atlas/Knee/Source/16/left/points.txt'

    ###fusion of landmarks####
    print('Running NCC_point_fusion_v2')
    movingimages = registered_atlas_images
    nr_atlas = 2
    kernel = 2
    NCC_point_fusion_v2(fixed_image, registered_atlas_images, outputfolder, mask_folder, nr_atlas, atlas_list,
                        vv_template_path, kernel)

    print('Multi-atlas segmentation of  completed in :', time_format)


def run_MAS(general_path, fixed_image_path, joint, side, outputfolder, radius=3):
    total_start = time.time()
    print('Mutli-Atlas segmentation of ', fixed_image_path, 'has started')
    # Define path to parameter file

    rigid = general_path + '/Scripts/Param_Files/Euler_Without_Mask_MSD.txt'
    affine = general_path + '/Scripts/Param_Files/Affine_Without_Mask_MSD.txt'
    bspline = general_path + '/Scripts/Param_Files/BSpline_Without_Mask_MSD.txt'
    bspline_inverse = general_path + '/Scripts/Param_Files/BSpline_Without_Mask_INVERSE.txt'

    # Define path to label fusion parameter file
    label_fusion_parameter_file_path = 'label_fusion.txt'

    # general_path='/Volumes/Luca_T5/4D_MSK'
    # atlas_folder=f'{general_path}/Atlas'
    atlas_folder = f'{general_path}/Atlas/' + joint + '/Source/'

    print('atlas_folder',atlas_folder)
    # side='right'

    # Define output directory
    # output_directory = f'{general_path}/Atlas/{subject}'
    output_directory = outputfolder

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load fixed image
    fixed_image = sitk.ReadImage(fixed_image_path)

    ## call function that creates dictionaries from atlas folder
    if joint == 'Ankle':
        atlas_data_paths = atlas_labels_dicts_ankle(atlas_folder, side)

    else:
        atlas_data_paths = atlas_labels_dicts_Knee(atlas_folder, side)
        print(atlas_data_paths)
    # Load atlas images, labels and landmarks
    atlas_images = []
    label_images = []
    anat_landmarks = []

    # Set up registration
    transformixImageFilter = sitk.TransformixImageFilter()
    registration = sitk.ElastixImageFilter()
    registration.SetFixedImage(fixed_image)
    registration.SetParameterMap(registration.ReadParameterFile(rigid))
    registration.AddParameterMap(registration.ReadParameterFile(affine))
    registration.AddParameterMap(registration.ReadParameterFile(bspline))

    transformixImageFilter.LogToConsoleOff()
    registration.LogToConsoleOn()

    # Set up inverse registration
    registration_inverse = sitk.ElastixImageFilter()
    registration_inverse.SetFixedImage(fixed_image)
    registration_inverse.SetParameterMap(registration.ReadParameterFile(bspline_inverse))
    registration_inverse.LogToConsoleOff()

    # Register each atlas to the fixed image
    registered_atlas_images = []
    transformed_labels = []
    transformed_landmarks = []
    atlas_list = []
    outputfolder_list = []

    for files in tqdm(atlas_data_paths):
        print('Running atlas : ', files['image'], ' Fixed image : ', fixed_image_path)
        atlas_image = sitk.ReadImage(files['image'])
        label_image = sitk.ReadImage(files['label'])
        landmarks = files['landmarks']
        atlas_images.append(atlas_image)
        label_images.append(label_image)
        anat_landmarks.append(landmarks)
        # registration.LogToConsoleOn()

        registration.SetMovingImage(atlas_image)
        folder_name = files['image'].split('/')[8]



        atlas_list.append(folder_name)

        output_atlas_folder = output_directory + '/' + folder_name
        outputfolder_list.append(output_atlas_folder)
        if not os.path.exists(output_atlas_folder):
            os.makedirs(output_atlas_folder)

        registration.SetOutputDirectory(output_atlas_folder)

        try:
            registration.Execute()
        except:
            continue
        registered_atlas_image = registration.GetResultImage()
        registered_atlas_images.append(registered_atlas_image)

        parameterMap = registration.GetTransformParameterMap()
        parameterMap_landmarks = registration.GetTransformParameterMap()

        parameterMap[2]['FinalBSplineInterpolationOrder'] = ['0']
        parameterMap[2]['DefaultPixelValue'] = ['0']
        parameterMap[2]['ResultImagePixelType'] = ['unsigned char']

        transformixImageFilter.SetMovingImage(label_image)
        transformixImageFilter.SetTransformParameterMap(parameterMap)
        transformixImageFilter.Execute()
        transformed_label = transformixImageFilter.GetResultImage()

        # cast to unchar
        mask = sitk.Cast(transformed_label, sitk.sitkUInt8)
        transformed_labels.append(mask)
        transformed_mask = output_atlas_folder + '/transformed_mask.nii'

        # if debug: save result for each transformed label
        sitk.WriteImage(mask, transformed_mask)

        # inverse
        transformParameterFile = os.path.join(output_atlas_folder, 'TransformParameters.2.txt')
        print('InitialTransformParameterFileName', transformParameterFile)
        registration_inverse.SetMovingImage(fixed_image)
        registration_inverse.SetInitialTransformParameterFileName(transformParameterFile)

        output_atlas_folder_inverse = output_atlas_folder + '/inverse'
        if not os.path.exists(output_atlas_folder_inverse):
            os.makedirs(output_atlas_folder_inverse)

        registration_inverse.SetOutputDirectory(output_atlas_folder_inverse)
        registration_inverse.Execute()
        parameterMap_inverse = registration_inverse.GetTransformParameterMap()
        parameterMap_inverse[0]['InitialTransformParametersFileName'] = ['NoInitialTransform']

        ## transform landmarks using the inverse
        transformixImageFilter.SetTransformParameterMap(parameterMap_inverse)
        transformixImageFilter.SetFixedPointSetFileName(landmarks)
        transformixImageFilter.SetOutputDirectory(output_atlas_folder)
        transformixImageFilter.Execute()

        ### extract transformed landmarks from outputpoints.txt
        transformix_pts_path = output_atlas_folder + '/outputpoints.txt'
        outputpath = output_atlas_folder + '/outputpoints_vv.txt'
        tfx_pts_file = output_atlas_folder + '/landmarks_tfx_' + folder_name + '.txt'
        vv_template_path = '/home/pieter/Documents/4D_MSK/Atlas/Knee/Source/16/left/points.txt'

        print('creating output points vv')
        raw_tfx_pts = create_vv_pts_from_outputtfx(vv_template_path, transformix_pts_path, outputpath, tfx_pts_file)

    #fusionfilter = sitk.MultiLabelSTAPLEImageFilter()
    #fusionfilter.SetLabelForUndecidedPixels(0)

    #segmentation_image = fusionfilter.Execute(transformed_labels)
    mask_folder = outputfolder + '/Mask'
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    #seg_output = mask_folder + '/segmentation.nii'
    # Save segmentation image
    #sitk.WriteImage(segmentation_image, seg_output)
    #print("Splitting segmentations into individual bones")


    duration = time.time() - total_start

    time_format = time.strftime("%H:%M:%S", time.gmtime(duration))
    print('Writing results to ', outputfolder)
    vv_template_path = '/home/pieter/Documents/4D_MSK/Atlas/Knee/Source/16/left/points.txt'

    ###fusion of landmarks####
    print('Running NCC_point_fusion_v2')
    movingimages = registered_atlas_images
    nr_atlas = 2
    kernel = 2
    NCC_point_fusion_v2(fixed_image, registered_atlas_images, outputfolder, mask_folder, nr_atlas, atlas_list,
                        vv_template_path, kernel)

    print('Multi-atlas segmentation of  completed in :', time_format)

def roundThousand(x):
    y = int(1000.0 * x + 0.5)
    return str(float(y) * .001)


def elapsedTime(start_time):
    dt = roundThousand(time.perf_counter() - start_time)
    print("    ", dt, "seconds")


def Segmentation_2_Mesh(sitk_image, smoothness, outputpath):
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    vtkimg = sitk2vtk.sitk2vtk(sitk_image)
    mesh = vtkutils.extractSurface(vtkimg, 1)
    mesh2 = vtkutils.cleanMesh(mesh, False)
    mesh3 = vtkutils.smoothMesh(mesh2, smoothness)
    vtkutils.writeMesh(mesh3, outputpath)


def transform_label(label, general_path, output_folder, transformParameterFile, fixed_image, smoothness=20):
    t = time.perf_counter()
    if isinstance(label, str):
        label_image = sitk.ReadImage(label)

    rigid_inverse = general_path + '/Scripts/Param_Files/Euler_Without_Mask_INVERSE.txt'

    transformixImageFilter = sitk.TransformixImageFilter()

    parameterMap = sitk.ReadParameterFile(transformParameterFile)
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)

    # Set up inverse registration
    registration_inverse = sitk.ElastixImageFilter()
    registration_inverse.SetFixedImage(fixed_image)
    registration_inverse.SetMovingImage(fixed_image)
    registration_inverse.SetInitialTransformParameterFileName(transformParameterFile)
    registration_inverse.SetParameterMap(registration_inverse.ReadParameterFile(rigid_inverse))
    registration_inverse.LogToConsoleOff()

    # inverse
    registration_inverse.Execute()
    parameterMap_inverse = registration_inverse.GetTransformParameterMap()
    parameterMap_inverse[0]['InitialTransformParametersFileName'] = ['NoInitialTransform']
    parameterMap_inverse[0]['FinalBSplineInterpolationOrder'] = ['0']
    parameterMap_inverse[0]['DefaultPixelValue'] = ['0']
    parameterMap_inverse[0]['ResultImagePixelType'] = ['unsigned char']

    ## transform label using the inverse
    transformixImageFilter.SetTransformParameterMap(parameterMap_inverse)
    transformixImageFilter.SetMovingImage(label_image)
    transformixImageFilter.SetOutputDirectory(output_folder)
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.Execute()

    transformed_label = transformixImageFilter.GetResultImage()

    # cast to unchar
    mask = sitk.Cast(transformed_label, sitk.sitkUInt8)
    # transformed_labels.append(mask)
    timept = output_folder.split('/')[-1]
    transformed_mask = output_folder + '/transformed_mask_' + timept + '.nii'
    outputpath_stl = output_folder + '/transformed_mask_' + timept + '.stl'
    outputpath_ply = output_folder + '/transformed_mask_' + timept + '.ply'

    # if debug: save result for each transformed label
    sitk.WriteImage(mask, transformed_mask)
    Segmentation_2_Mesh(mask, smoothness, outputpath_stl)
    Segmentation_2_Mesh(mask, smoothness, outputpath_ply)

    elapsedTime(t)

# transform landmarks
def Transform_bony_landmarks_knee(path_reg, mask_names):
    '''Path_reg = path to the dynamic msk folder of the subject. i.e folder which has the "Input
    and "Output" folders

    '''

    header = 1
    points_path = f'{path_reg}/points/points.txt'
    points = np.genfromtxt(points_path, skip_header=header, filling_values=np.NaN)
    Femur_from_points = points[0:9, :]
    Extra_femur_points = points[22:25, :]
    Femur_from_points = np.vstack((Femur_from_points, Extra_femur_points))
    Tibia_from_points = points[9:15, :]
    Patella_from_points = points[15:22, :]

    bone_landmarks = [Femur_from_points, Tibia_from_points, Patella_from_points]

    for count, landmark in enumerate(bone_landmarks):
        print('Transforming ', mask_names[count], ' landmarks  ')
        # print(landmark)
        transform_list = sorted(glob.glob(f'{path_reg}/Output/{mask_names[count]}/Transform/*.txt'))

        joint = mask_names[count].split('_')[0]


        outputpts_path = f'{path_reg}/points/' + joint + '_mypts.csv'
        execute_transform_points_v2(landmark, outputpts_path, transform_list)

def Sequential_reg_elastix_Knee(Dynamic_MSK_folder, parameter_file_path, sequence='zyx', debug=False, Verbose=False):
    def binary_resample(ref, img):
        """
        Resamples an input image (binary) using a reference image.

        Parameters
        ----------
        ref : SimpleITK.Image

        img : SimpleITK.Image

        Returns
        -------
        resampled : SimpleITK.Image
        """
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetReferenceImage(ref)
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        resample_filter.SetOutputPixelType(img.GetPixelID())
        resampled = resample_filter.Execute(img)

        return resampled

    def create_directories(output_directory, num_moving_images):
        for i in range(0, num_moving_images):
            moving_image_no = "%02d" % (i)
            output_folder = os.path.join(output_directory, moving_image_no)
            # print(output_folder)
            try:
                os.makedirs(output_folder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

    # Dynamic_MSK_folder='/Volumes/Luca_T5/4D_MSK/Dynamic_data/Knee/Input/30/Test1/left/Straight_To_Bend'
    moving_image_paths = sorted(glob.glob(Dynamic_MSK_folder + '/Moving/*.mhd'))

    fixed_masks_path = sorted(glob.glob(Dynamic_MSK_folder + '/Mask/*_r3.mhd'))

    fixed_image_path = sorted(glob.glob(Dynamic_MSK_folder + '/Fixed/*.mhd'))

    # Load fixed image
    fixed_image = fixed_image_path[0]

    print('fixed Image',fixed_image )

    fixed_tpt = fixed_image.split('/')[-1].split('p')[1].split('.')[0]
    print('Fixed image', fixed_tpt)
    fixed_tpt = int(fixed_tpt)

    # Load moving images
    moving_images = []
    if Verbose:
        print('sorting out moving images')
    else:
        pass

    for moving_path in moving_image_paths:
        #     print('moving_path',moving_path)
        # moving_image = sitk.ReadImage(moving_path)
        moving_image = moving_path
        moving_images.append(moving_image)

    fixed_masks_path.reverse()
    fixed_mask_images = []
    mask_names = []
    mask_paths = []

    for fixed_masks in fixed_masks_path:
        mask_name = fixed_masks.split('/')[-1].split('.')[0]
        if Verbose:
            print('fixed_masks', mask_name)
        else:
            pass

        fixed_mask = sitk.ReadImage(fixed_masks)
        fixed_mask = binary_resample(sitk.ReadImage(fixed_image), fixed_mask)

        sitk.WriteImage(fixed_mask, fixed_masks)
        fixed_mask_images.append(fixed_masks)
        mask_names.append(mask_name)

    # Set up registration
    selx = sitk.ElastixImageFilter()
    stfx = sitk.TransformixImageFilter()

    mask_no = 0
    for mask in tqdm(fixed_mask_images):
        ##perform first registrattion between fixed image and tthe target image (first moving image)
        # target image is the index of the moving images list for e.g '0' represents the first moving image in the list
        if Verbose:
            print('Initialise registrations')
        else:
            pass

        # target_image = fixed_tpt  #
        target_image = 0
        moving_image_no = "%02d" % (target_image)

        # Define output directory and create
        output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
        num_moving_images = len(moving_image_paths)
        if Verbose:
            print('number of moving images: ', num_moving_images)
        else:
            pass

        create_directories(output_directory, num_moving_images)

        if mask_no == 1:
            mask_init = mask_no - 1
            output_directory_init = Dynamic_MSK_folder + '/Output/' + mask_names[mask_init]
            temp = os.path.join(output_directory_init, moving_image_no)
            pathInitTrans = os.path.join(temp, 'TransformParameters.0.txt')

            output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
            output_result_folder = os.path.join(output_directory, moving_image_no)
            moving = moving_images[target_image]

            if Verbose:
                print('Initial transform parameter file : ', pathInitTrans)
                print('Fixed image                      : ', fixed_image)
                print('Moving image                     : ', moving)
                print('mask image                       : ', mask)
                print('Outputfolder                     : ', output_result_folder)

                subprocess.run(['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask,'-p', parameter_file_path, '-out',
                                output_result_folder, '-t0',pathInitTrans])

            else:
                subprocess.run(
                    ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                     output_result_folder, '-t0', pathInitTrans])




        else:
            output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
            output_result_folder = os.path.join(output_directory, moving_image_no)
            moving = moving_images[target_image]

            if Verbose:
                # print('Initial transform parameter file : ', pathInitTrans )
                print('Fixed image                      : ', fixed_image)
                print('Moving image                     : ', moving)
                print('mask image                       : ', mask)
                print('Outputfolder                     : ', output_result_folder)


                subprocess.run(
                    ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                     output_result_folder])

            else:
                subprocess.run(
                    ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                     output_result_folder])


        movingImage = moving_image_paths[target_image]
        movingImage_name = movingImage.split('/')[-1].split('.')[0]

        # upward ### remember it uses 0 indexing

        while target_image > 0:
            if Verbose:
                print('DOWNWARD')
            else:
                pass

            output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
            if mask_no == 1:
                mask_init = mask_no - 1
                output_directory_init = Dynamic_MSK_folder + '/Output/' + mask_names[mask_init]
                temp = os.path.join(output_directory_init, moving_image_no)
                pathInitTrans = os.path.join(temp, 'TransformParameters.0.txt')

                output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
                output_result_folder = os.path.join(output_directory, moving_image_no)
                moving = moving_images[target_image]

                if Verbose:
                    print('Initial transform parameter file : ', pathInitTrans)
                    print('Fixed image                      : ', fixed_image)
                    print('Moving image                     : ', moving)
                    print('mask image                       : ', mask)
                    print('Outputfolder                     : ', output_result_folder)

                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder,'-t0', pathInitTrans])

                else:
                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder, '-t0', pathInitTrans])



            else:
                target_image = target_image - 1
                initial_tf_no = target_image + 1
                init_moving_image_no = "%02d" % (initial_tf_no)

                # print('init_moving_image_no', init_moving_image_no)
                moving_image_no = "%02d" % (target_image)
                moving = moving_images[target_image]
                print('moving', moving)

                temp = os.path.join(output_directory, init_moving_image_no)
                pathInitTrans = os.path.join(temp, 'TransformParameters.0.txt')
                output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
                output_result_folder = os.path.join(output_directory, moving_image_no)

                if Verbose:
                    print('Initial transform parameter file : ', pathInitTrans)
                    print('Fixed image                      : ', fixed_image)
                    print('Moving image                     : ', moving)
                    print('mask image                       : ', mask)
                    print('Outputfolder                     : ', output_result_folder)

                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder, '-t0', pathInitTrans])

                else:
                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder, '-t0', pathInitTrans])


        while target_image < num_moving_images - 1:
            if Verbose:
                print('UPWARD')
            else:
                pass

            target_image = target_image + 1
            initial_tf_no = target_image - 1
            #     print('target_image',target_image)
            # print('TARGET',target_image)

            init_moving_image_no = "%02d" % (initial_tf_no)
            moving_image_no = "%02d" % (target_image)
            moving = moving_images[target_image]

            # print('init_moving_image_no', init_moving_image_no)
            if mask_no == 1:
                mask_init = mask_no - 1
                output_directory_init = Dynamic_MSK_folder + '/Output/' + mask_names[mask_init]
                temp = os.path.join(output_directory_init, moving_image_no)
                pathInitTrans = os.path.join(temp, 'TransformParameters.0.txt')

                output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
                output_result_folder = os.path.join(output_directory, moving_image_no)

                if Verbose:
                    print('Initial transform parameter file : ', pathInitTrans)
                    print('Fixed image                      : ', fixed_image)
                    print('Moving image                     : ', moving)
                    print('mask image                       : ', mask)
                    print('Outputfolder                     : ', output_result_folder)

                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder, '-t0', pathInitTrans])


                else:
                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder, '-t0', pathInitTrans])


            else:

                temp = os.path.join(output_directory, init_moving_image_no)
                pathInitTrans = os.path.join(temp, 'TransformParameters.0.txt')
                output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]

                output_directory = Dynamic_MSK_folder + '/Output/' + mask_names[mask_no]
                output_result_folder = os.path.join(output_directory, moving_image_no)

                if Verbose:
                    print('Initial transform parameter file : ', pathInitTrans)
                    print('Fixed image                      : ', fixed_image)
                    print('Moving image                     : ', moving)
                    print('mask image                       : ', mask)
                    print('Outputfolder                     : ', output_result_folder)

                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder, '-t0', pathInitTrans])


                else:
                    subprocess.run(
                        ['elastix', '-f', fixed_image, '-m', moving, '-fMask', mask, '-p', parameter_file_path, '-out',
                         output_result_folder, '-t0', pathInitTrans])

        mask_no = mask_no + 1
    fixed_pos = [fixed_tpt]

    mask_names=['Femur_final_r3','Tibia_final_r3','Patella_final_r3']
    relative_Tx_of_MC1, sequential_Tx = Calculate_cardan_angles_Knee(Dynamic_MSK_folder, fixed_pos, mask_names,
                                                                     sequence=sequence)
    Transform_bony_landmarks_knee(Dynamic_MSK_folder,mask_names)

# MAC
# Dynamic_MSK_folder='/Volumes/BK_T5/4D_MSK/Dynamic_data/Knee/Input/23/Test1/left/Straight_To_Bend'
# parameter_file_path ='/Volumes/BK_T5/4D_MSK/Scripts/Param_Files/Parameter_File_Mathias_MSD.txt'

# import time

# total_start = time.time()


# # Sequential_reg_Knee(Dynamic_MSK_folder,parameter_file_path,Verbose=True)
# Sequential_reg_elastix_Knee(Dynamic_MSK_folder,parameter_file_path,Verbose=True)

# duration=time.time() - total_start

# time_format = time.strftime("%H:%M:%S", time.gmtime(duration))

# print('PROCESSING OF 4DMSK IMAGES FOR SUBJECT COMPLETED IN :' ,time_format)

# Run inference on sh data using the trained model



def RunMyInference(traindir, model, output_dir):
    total_start = time.time()

    mhdImage = sorted(glob.glob(traindir + '/*.mhd'))
    ImageName = mhdImage[0].split("/")[-1].split(".")[0]
    niftyImage = traindir + '/' + ImageName + '.nii'
    tempIm = sitk.ReadImage(mhdImage[0])
    sitk.WriteImage(tempIm, niftyImage)

    def Refine_segmentation_knee(inputdatapath, outputdata, radius):

        kernel = (radius, radius, radius)
        inputdata = sitk.ReadImage(inputdatapath)

        # original segmented images
        Tibia_og = sitk.BinaryThreshold(inputdata, 1, 1)
        femur_og = sitk.BinaryThreshold(inputdata, 2, 2)
        patella_og = sitk.BinaryThreshold(inputdata, 3, 3)

        Tibiapath_og = outputdata + '/Tibia_final.mhd'
        femurpath_og = outputdata + '/Femur_final.mhd'
        patellapath_og = outputdata + '/Patella_final.mhd'

        # binarize images
        Tibia = sitk.BinaryThreshold(inputdata, 1, 1)
        Femur = sitk.BinaryThreshold(inputdata, 2, 2)
        Patella = sitk.BinaryThreshold(inputdata, 3, 3)
        # Femur = sitk.BinaryThreshold(inputdata, Thresh3, Thresh3)

        # morphological image closing
        Tibia = sitk.BinaryMorphologicalClosing(Tibia, kernel)
        Femur = sitk.BinaryMorphologicalClosing(Femur, kernel)
        Patella = sitk.BinaryMorphologicalClosing(Patella, kernel)
        # Femur = sitk.BinaryMorphologicalClosing(Femur, kernel)

        # morphological dilation
        Tibia = sitk.BinaryDilate(Tibia, kernel)
        Femur = sitk.BinaryDilate(Femur, kernel)
        Patella = sitk.BinaryDilate(Patella, kernel)
        # Femur = sitk.BinaryDilate(Femur, kernel)

        Tibiapath = outputdata + '/Tibia_final_r' + str(radius) + '.mhd'
        Femurpath = outputdata + '/Femur_final_r' + str(radius) + '.mhd'
        Patellapath = outputdata + '/Patella_final_r' + str(radius) + '.mhd'
        # femurpath = outputdata + '/Femur_final_r' + str(radius) + '.mhd'

        sitk.WriteImage(Tibia, Tibiapath)
        sitk.WriteImage(Femur, Femurpath)
        sitk.WriteImage(Patella, Patellapath)
        # sitk.WriteImage(Femur, femurpath)

    # traindir='/media/pieter/Luca_T5/4D_MSK/UNET/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task101_MSK'
    def validate_dicts(traindir):

        val_T1_dir = sorted(glob.glob(traindir + '/*.nii'))
        val_labels_dir = sorted(glob.glob(traindir + '/labels/*.nii'))

        val_files = [{'image': T1, 'label': label}
                     for T1, label in zip(val_T1_dir, val_T1_dir)]

        return val_files

    def create_transforms(mode='train', keys=("image", "label")):
        val_transforms = Compose([
            LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
            Spacingd(keys=['image', 'label'], pixdim=[1, 2, 4], mode=("bilinear", "nearest")),
            #             EnsureChannelFirstd(keys=['image','label']),
            # ThresholdIntensityd(keys=['image'],threshold=-1027, above=True, cval=0.0),

            # be carefull here should be the same as above
            # NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys=["image"], lower=1, upper=99,
                b_min=0.0, b_max=1.0, clip=True,
            ), ]
            # ToTensord(keys=['image', 'label'])]

        )
        post_transforms = Compose(
            [
                Invertd(
                    keys="pred",
                    transform=val_transforms,
                    orig_keys="label",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="label_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                    device="cpu",
                ),
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True),
                KeepLargestConnectedComponentd(keys="pred", num_components=1),
                SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg",
                           resample=False),
            ]
        )

        return Compose(val_transforms), post_transforms

    in_channels = 1
    n_class = 4
    spacings = [1, 2, 4]
    strides, kernels = [], []
    # sizes=[256, 256,50]
    sizes = [128, 64, 32]

    device = torch.device("cuda:0")
    learning_rate = 0.001

    # The rule boiles down to the following
    # downsample each dimension with a stride of 2 and kernel size of 3 untill it reached it's bottleneck representation (smaller then 8)
    # then stop each channel individually but continue the others.

    while True:  # https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_pipeline/create_network.py
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    def inferer(image, model):
        roi_size = (256, 128, 64)  # should be twice the patch size
        sw_batch_size = 1
        return sliding_window_inference(image, roi_size, sw_batch_size, model, overlap=0.33)

    def infer(val_loader, model_name, post_transforms):
        dice_metric = DiceMetric(include_background=False, reduction='mean')
        print('running inference function')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_class,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            dropout=0,
            deep_supervision=True,
            deep_supr_num=2,
            res_block=True,
        ).to(device)

        #     model = SwinUNETR(
        #     img_size=(128,64,32),
        #     in_channels=1,
        #     out_channels=n_class,
        #     feature_size=48,
        #     use_checkpoint=True,
        #     ).to(device)

        print('model is created')
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage), strict=False)
        print('model is laoded')
        model.eval()
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=n_class)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=n_class)])

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                val_inputs = batch["image"].cuda()
                original_affine = batch["image_meta_dict"]["affine"][0].numpy()
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

                print("Inference on case {}".format(img_name))
                ct = 1.0
                val_outputs = inferer(val_inputs, model)

                #             for dims in [[0,1]]:
                #                     flip_inputs = torch.flip(val_inputs, dims=dims)
                #                     flip_outputs = inferer( flip_inputs, model)
                #                     flip_pred = torch.flip(flip_outputs, dims=dims)
                #                     flip_pred = torch.softmax(flip_pred, dim=1)
                #                     del flip_inputs
                #                     val_outputs += flip_pred
                #                     del flip_pred
                #                     ct += 1

                #             val_outputs=val_outputs/ct
                # val_outputs=val_outputs[0,1,:,:,:]
                batch["pred"] = val_outputs

                batch = [post_transforms(i) for i in decollate_batch(batch)]


    val_files = validate_dicts(traindir)
    val_files = val_files
    val_transforms, post_transforms = create_transforms(mode='infer', keys=("image",))
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    print('loaders are created')
    infer(val_loader, model, post_transforms)

    duration = time.time() - total_start

    time_format = time.strftime("%H:%M:%S", time.gmtime(duration))

    print('Splitting segmentation into individual bones')
    seg = sorted(glob.glob(output_dir + '/' + ImageName + '/*.gz'))
    radius = 3

    Refine_segmentation_knee(seg[0], output_dir, radius)

    print('total duration is ', time_format)


def run_4DMSK_processing(general_path, path_dicom_static, path_dicom_dynamic,
                         joint, side, subject, fixed_tpt, output_folder=None, split=True, bipedal=True):
    total_start = time.time()
    if bipedal == True:
        split_static = True
        split_dyn = True
    else:
        split_static = True
        split_dyn = False

    if output_folder == None:
        output_folder_static = os.path.join(general_path, subject, 'Static')
        output_folder_dynamic = os.path.join(general_path, subject)
    else:
        output_folder_static = os.path.join(output_folder, subject, 'Static')
        output_folder_dynamic = os.path.join(output_folder, subject)

    print('PROCESSING OF 4DMSK IMAGES FOR SUBJECT  ', subject, 'HAS STARTED')

    ####################################### convert dicom to mhd ######################################
    print('********** ********** converting dicom images to MHD files ********** ********** ')
    Path_to_dynamic_images = read_DCM_to_MHD(path_dicom_dynamic, output_folder_dynamic)
    Path_to_static_image = read_DCM_to_MHD(path_dicom_static, output_folder_static)
    ####################################### convert dicom to mhd ######################################

    ####################################### preprocess and split dynamic images ######################################
    print('********** ********** pre-processing dynamic MHD images ********** ********** ')
    folder1, folder2 = pre_process_images(Path_to_dynamic_images, output_folder_dynamic, side, split_dyn, static=False)

    print('********** ********** pre-processing static MHD images   ********** ********** ')
    static_folder1, static_folder2 = pre_process_images(Path_to_static_image, output_folder_static, side, split_static,
                                                        static=True)
    ####################################### preprocess and split dynamic images ######################################

    print('folder1', folder1)
    print('folder2', folder2)

    moving = os.path.join(folder1, 'Moving')
    # print(output_folder)
    try:
        os.makedirs(moving)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # fetch all files
    for file_name in os.listdir(folder1):
        # construct full file path
        source = os.path.join(folder1, file_name)
        destination = os.path.join(moving, file_name)
        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)
            # print('Moved:', file_name)

    moving_image_paths = sorted(glob.glob(moving + '/*.mhd'))

    fixed = os.path.join(folder1, 'Fixed')
    # print(output_folder)
    try:
        os.makedirs(fixed)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    fixed_path_mhd = moving_image_paths[fixed_tpt]

    sitk_fixed = sitk.ReadImage(fixed_path_mhd)
    path_fixed = os.path.join(fixed, 's01p00.mhd')
    sitk.WriteImage(sitk_fixed, path_fixed)

    outputfolder_mask = os.path.join(folder1, 'Mask')
    # print(output_folder)
    try:
        os.makedirs(outputfolder_mask)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    #

    ###################################### Determine Fixed image######################################

    ###################################### Multi-atlas segmentation ######################################
    # run_MAS (general_path,moving_image_paths[int(fixed_tpt)],joint,side,subject,outputfolder,radius)

    print('joint', joint)
    run_MAS(general_path, path_fixed, joint, side, subject, outputfolder_mask, radius)
    ###################################### Multi-atlas segmentation ######################################

    # ###################################### Sequentially register images ######################################
    parameter_file_path = '/Volumes/Luca_T5/4D_MSK/Scripts/Param_Files/Parameter_File_Mathias_MSD_edit.txt'
    Sequential_reg(folder1, fixed_tpt, parameter_file_path, debug=False, Verbose=True)
    # Sequential_reg(folder2,parameter_file_path,debug=False,Verbose=True)
    # ###################################### Sequentially register images ######################################

    duration = time.time() - total_start

    time_format = time.strftime("%H:%M:%S", time.gmtime(duration))

    print('PROCESSING OF 4DMSK IMAGES FOR SUBJECT  ', subject, ' COMPLETED IN :', time_format)


# Linux
if __name__ == '__main__':

    #Do not change
    general_path = '/home/pieter/Documents/4D_MSK'
    model = '/home/pieter/Documents/4D_MSK/Resources/DynUNet-MSK_0.001_Largebest'
    parameter_file_path = '/home/pieter/Documents/4D_MSK/Scripts/Param_Files/Parameter_File_Mathias_MSD.txt'
    point = 'points'
    conditions = ['Knee']
    joint = 'Knee'


    #Can be changed
    sides = ['left']
    subjects = ['HS_001']
    Tests = ['Baseline']
    # use when you want to do the ****ACTUAL**** processing and the script will save the results elsewehere
    # angle_maxs = [-15,-20,-25,-30,-35]

    # use if you want to graph different angles and different subjects
    angle_maxs = [-35]

    import time

    total_start = time.time()
    for test in Tests:
        for subject in subjects:
            for side in sides:
                Dynamic_MSK_folder =f'/Input/{test}/{subject}/{side}'
                outputfolder = Dynamic_MSK_folder + '/Mask'
                print('outputfolder', outputfolder)
                # parameter_file_path ='/media/pieter/BK_T5/4D_MSK/Atlas/Knee/Scripts/Param_Files/Euler_With_Mask.txt'
                # parameter_file_path ='/home/pieter/Documents/4D_MSK/Scripts/Param_Files/Parameter_File_Mathias_MI.txt'

                print('Segmenting bones of interest')
                traindir = Dynamic_MSK_folder+'/Fixed'
                output_dir = Dynamic_MSK_folder+'/Mask'


                #Model for segmentation
                # RunMyInference(traindir, model, output_dir)

                #Run MAS but just to obtain landmark propogation
                fixed_image_path=sorted(glob.glob(traindir + '/*.mhd'))
                # run_MAS_ELASTIX(general_path, fixed_image_path[0], joint, side, outputfolder, radius=3)

                print('Sequential registration of bones of interest')
                Sequential_reg_elastix_Knee(Dynamic_MSK_folder,parameter_file_path,sequence='zyx',Verbose=True)

                # Orthopeadic metric computation
                Compute_Orthopedic_Metrics(subjects, conditions, Tests, sides, angle_maxs, point)

                # run_4DMSK_processing(general_path, path_dicom_static, path_dicom_dynamic,
                #                      joint, side, subject, outputfolder, split)

