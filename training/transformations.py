import json
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

def augment_from_config(config_path: str):
    """Load augmentation parameters from a JSON config and build an Albumentations Compose."""
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    inp_size = tuple(cfg["input_size"])

    # Adds channel dimension
    add_channels = A.Lambda(
        image=lambda x, **kwargs: x[..., np.newaxis]
    )

    # Elastic deformation
    elastic_cfg = cfg["elastic"]
    elastic = A.ElasticTransform(
        alpha=elastic_cfg["alpha"],
        sigma=elastic_cfg["sigma"],
        interpolation=getattr(cv2, elastic_cfg["interpolation"]),
        p=elastic_cfg["p"]
    )

    # Speckle noise
    speckle_cfg = cfg["speckle"]
    speckle = A.MultiplicativeNoise(
        multiplier=tuple(speckle_cfg["multiplier"]),
        elementwise=speckle_cfg["elementwise"],
        p=speckle_cfg["p"]
    )

    # Compose the full augmentation pipeline
    return A.Compose(
        [
            add_channels,
            A.Resize(inp_size[0], inp_size[1],
                     interpolation=getattr(cv2, cfg["resize"]["interpolation"]),
                     p=cfg["resize"]["p"]),
            A.HorizontalFlip(p=cfg["horizontal_flip"]["p"]),
            A.ShiftScaleRotate(
                shift_limit=cfg["ssr"]["shift_limit"],
                scale_limit=cfg["ssr"]["scale_limit"],
                rotate_limit=cfg["ssr"]["rotate_limit"],
                border_mode=cfg["ssr"]["border_mode"],
                interpolation=cfg["ssr"]["interpolation"],
                p=cfg["ssr"]["p"]
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg["brightness_contrast"]["brightness_limit"],
                contrast_limit=cfg["brightness_contrast"]["contrast_limit"],
                p=cfg["brightness_contrast"]["p"]
            ),
            elastic,
            speckle,
            A.Normalize(mean=tuple(cfg["normalize"]["mean"]),
                        std=tuple(cfg["normalize"]["std"]),
                        max_pixel_value=cfg["normalize"]["max_pixel_value"]),
            ToTensorV2(transpose_mask=cfg["to_tensor"]["transpose_mask"]),
        ],
        additional_targets=cfg["additional_targets"]
    )


def default_augmentation(inp_size=(512, 512)):
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