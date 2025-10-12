import torch
import numpy as np
from scipy import ndimage
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


####################
# Basic Transforms #
####################

def resize_arr(arr, out_shape, order=1):
    '''Resizes numpy array to out_shape with given order interpolation.'''

    in_H, in_W = arr.shape
    out_H, out_W = out_shape
    zoom_factors = (out_H / in_H, out_W / in_W)
    return ndimage.zoom(arr, zoom=zoom_factors, order=order)


def img2input(arr, inp_size=(512, 512)):
    '''
    Resizes numpy array to inp_size, rescales to [0, 1],
    adds batch and channel dimensions, and converts to a
    torch tensor. This is typically the final transform
    that needs to be applied.
    '''

    resized = resize_arr(arr, inp_size)
    resized = resized.astype(np.float32) / 255.0
    resized = resized[np.newaxis, np.newaxis, :, :]
    return torch.from_numpy(resized)


def label2input(arr, inp_size=(512, 512)):
    '''
    Resizes numpy array to inp_size via nearest neighbors,
    adds batch and channel dimensions, and converts to a
    torch tensor. This is typically the final transform
    that needs to be applied.
    '''

    resized = resize_arr(arr, inp_size, order=0)
    resized = resized.astype(np.float32)
    resized = resized[np.newaxis, np.newaxis, :, :]
    return torch.from_numpy(resized)


######################
# Data Augmentations #
######################

def augment_from_config(inp_size=(512, 512)):

    # adds channel dimension
    add_channels = A.Lambda(
        image=lambda x, **kwargs: x[..., np.newaxis]
    )

    # elastic deformation
    elastic = A.ElasticTransform(
        alpha=20,           # intensity of deformation
        sigma=6,            # smoothing of displacement
        alpha_affine=10,    # optional affine component
        interpolation=1,    # cv2.INTER_LINEAR for images
        p=0.5
    )

    # speckle
    speckle = A.MultiplicativeNoise(
        multiplier=(0.9, 1.1),  # multiplicative range
        elementwise=True,       # vary pixel-by-pixel (speckle-like)
        p=0.5
    )

    return A.Compose(
        [
            add_channels,
            A.Resize(inp_size[0], inp_size[1], interpolation=cv2.INTER_LINEAR, p=1.0),
            A.HorizontalFlip(p=0.5),

            # Mild geometry + photometrics
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                               border_mode=0, value=0, mask_value=0, interpolation=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

            # elastic deformations and speckle
            elastic,
            speckle,

            # Convert from uint8 to float32 and tensor
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
            ToTensorV2(transpose_mask=True),
        ],
        # IMPORTANT: this tells Albumentations that "label" is a mask so it uses nearest-neighbor interp
        additional_targets={'label': 'mask'},
    )

def default_augmentation(inp_size=(512, 512)) :
    # adds channel dimension
    add_channels = A.Lambda(
        image=lambda x, **kwargs: x[..., np.newaxis]
    )

    return A.Compose(
        [
            add_channels,
            A.Resize(inp_size[0], inp_size[1], interpolation=cv2.INTER_LINEAR, p=1.0),

            # Convert from uint8 to float32 and tensor
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
            ToTensorV2(transpose_mask=True),
        ],
        # IMPORTANT: this tells Albumentations that "label" is a mask so it uses nearest-neighbor interp
        additional_targets={'label': 'mask'},
    )