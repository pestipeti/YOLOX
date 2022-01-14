#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh

import math
import random
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.functional import MAX_VALUES_BY_DTYPE
from functools import wraps


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                value
            )
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(img.dtype)
            )
    return img.astype("float32") / max_value


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).astype(dtype)


def preserve_shape(func):
    """
    Preserve shape of the image

    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


@preserve_shape
def add_rain(
    img,
    slant,
    drop_length,
    drop_width,
    drop_color,
    blur_value,
    brightness_coefficient,
    rain_drops,
):
    """

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        slant (int):
        drop_length:
        drop_width:
        drop_color:
        blur_value (int): Rainy view are blurry.
        brightness_coefficient (float): Rainy days are usually shady.
        rain_drops:

    Returns:
        numpy.ndarray: Image.

    """
    # non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomRain augmentation".format(input_dtype))

    image_o = img.copy()
    image = img.copy()
    image *= 0

    for (rain_drop_x0, rain_drop_y0) in rain_drops:
        rain_drop_x1 = rain_drop_x0 + slant
        rain_drop_y1 = rain_drop_y0 + drop_length

        cv2.line(
            image,
            (rain_drop_x0, rain_drop_y0),
            (rain_drop_x1, rain_drop_y1),
            drop_color,
            drop_width,
        )

    image = cv2.blur(image, (blur_value, blur_value))  # rainy view are blurry
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    image_hsv[:, :, 2] *= brightness_coefficient

    image_rgb = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    image_o[np.where(image_rgb != 0)] = image_rgb[np.where(image_rgb != 0)]

    if needs_float:
        image_o = to_float(image_o, max_value=255)

    return image_o


class RandomRain(ImageOnlyTransform):
    """Adds rain effects.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        slant_lower: should be in range [-20, 20].
        slant_upper: should be in range [-20, 20].
        drop_length: should be in range [0, 100].
        drop_width: should be in range [1, 5].
        drop_color (list of (r, g, b)): rain lines color.
        blur_value (int): rainy view are blurry
        brightness_coefficient (float): rainy days are usually shady. Should be in range [0, 1].
        rain_type: One of [None, "drizzle", "heavy", "torrestial"]

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        slant_lower=-10,
        slant_upper=10,
        drop_length=20,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=7,
        brightness_coefficient=0.7,
        rain_type=None,
        always_apply=False,
        p=0.5,
    ):
        super(RandomRain, self).__init__(always_apply, p)

        if rain_type not in ["drizzle", "heavy", "torrential", "fish", None]:
            raise ValueError(
                "raint_type must be one of ({}). Got: {}".format(["drizzle", "heavy", "torrential", None], rain_type)
            )
        if not -20 <= slant_lower <= slant_upper <= 20:
            raise ValueError(
                "Invalid combination of slant_lower and slant_upper. Got: {}".format((slant_lower, slant_upper))
            )
        if not 1 <= drop_width <= 5:
            raise ValueError("drop_width must be in range [1, 5]. Got: {}".format(drop_width))
        if not 0 <= drop_length <= 100:
            raise ValueError("drop_length must be in range [0, 100]. Got: {}".format(drop_length))
        if not 0 <= brightness_coefficient <= 1:
            raise ValueError("brightness_coefficient must be in range [0, 1]. Got: {}".format(brightness_coefficient))

        self.slant_lower = slant_lower
        self.slant_upper = slant_upper

        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.rain_type = rain_type

    def apply(self, image, slant=10, drop_length=20, rain_drops=(), **params):
        return add_rain(
            image,
            slant,
            drop_length,
            self.drop_width,
            self.drop_color,
            self.blur_value,
            self.brightness_coefficient,
            rain_drops,
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        slant = int(random.uniform(self.slant_lower, self.slant_upper))

        height, width = img.shape[:2]
        area = height * width

        if self.rain_type == "fish":
            num_drops = int(random.uniform(area / 770, area / 600))
            drop_length = int(random.uniform(5, 18))
            color = [0, 0, 0]
            for i in range(3):
                color[i] = self.drop_color[i] + int(random.uniform(-20, 20))

            self.drop_color = color

            if np.random.rand() < 0.1:
                self.drop_width = 2

        elif self.rain_type == "drizzle":
            num_drops = area // 770
            drop_length = 10
        elif self.rain_type == "heavy":
            num_drops = width * height // 600
            drop_length = 30
        elif self.rain_type == "torrential":
            num_drops = area // 500
            drop_length = 60
        else:
            drop_length = self.drop_length
            num_drops = area // 600

        rain_drops = []

        for _i in range(num_drops):  # If You want heavy rain, try increasing this
            if slant < 0:
                x = random.randint(slant, width)
            else:
                x = random.randint(0, width - slant)

            y = random.randint(0, height - drop_length)

            rain_drops.append((x, y))

        return {"drop_length": drop_length, "rain_drops": rain_drops, "slant": slant}

    def get_transform_init_args_names(self):
        return (
            "slant_lower",
            "slant_upper",
            "drop_length",
            "drop_width",
            "drop_color",
            "blur_value",
            "brightness_coefficient",
            "rain_type",
        )


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, albu=None):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.albu = albu.lower() if albu else None

        self.transformHigh = A.Compose([
            A.HorizontalFlip(),
            RandomRain(rain_type='fish', brightness_coefficient=1.0, blur_value=1, drop_color=(190, 120, 60), p=0.5),
            A.OneOf([
                A.RGBShift(p=0.9),
                A.ToGray(p=0.05),
                A.CLAHE(p=0.05),
            ], p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.1), p=0.75),
            A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.1, p=0.5),
            A.ShiftScaleRotate(scale_limit=(-0.2, 0.05), rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.75),

            A.GaussNoise(p=1.0),
            A.CoarseDropout(min_holes=24, max_holes=32, min_width=8, max_width=32, min_height=8, max_height=32, p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        self.transformMed = A.Compose([
            A.HorizontalFlip(),

            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=35, p=0.7),
            RandomRain(rain_type='fish', brightness_coefficient=1.0, blur_value=1, drop_color=(190, 120, 60), p=0.25),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0)),
                A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.1),
            ], p=0.5),

            A.OneOf([
                A.Perspective(),
                A.ShiftScaleRotate(scale_limit=(-0.4, 0), shift_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(scale_limit=(0, 0.2), shift_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(rotate_limit=20, shift_limit=0, scale_limit=0, border_mode=cv2.BORDER_CONSTANT),
            ], p=0.7),

            A.GaussNoise(p=0.7),

        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # Low augmentation
        self.transformLow = A.Compose([
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=15, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        self.transformNone = A.Compose([
            A.NoOp(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if self.albu == 'low':
            transform = self.transformLow
        elif self.albu == 'high':
            transform = self.transformHigh
        elif self.albu == 'med':
            transform = self.transformMed
        else:
            transform = self.transformNone

        # Albumentations
        new = transform(image=image, bboxes=boxes, class_labels=labels)  # transformed
        image_t = new["image"]
        labels = np.array(new["class_labels"])
        boxes = np.array([b for b in new["bboxes"]])

        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image_t, r_o = preproc(image_t, input_dim)
            return image_t, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        # if random.random() < self.hsv_prob:
        #     augment_hsv(image)
        image_t, boxes = _mirror(image_t, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)

        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # return img, np.zeros((1, 5))
        return img, res
