from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose, ToGray, InvertImg, HorizontalFlip
)

import cv2


class Augmentation:
    def __init__(self, strength=1.):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        self.strength = strength
        self.augs = self.get_augmentations()

    def __call__(self, data):
        if self.augs is not None:
            data = self.augs(**data)
        return data

    def get_photometric(self):
        coeff = int(3 * self.strength)
        k = max(1, coeff if coeff % 2 else coeff - 1)

        return Compose([
            OneOf([
                CLAHE(clip_limit=2, p=.4),
                IAASharpen(p=.3),
                IAAEmboss(p=.3),
            ], p=0.5),
            OneOf([
                IAAAdditiveGaussianNoise(p=.3),
                GaussNoise(p=.7),
            ], p=.5),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=k, p=.3),
                Blur(blur_limit=k, p=.5),
            ], p=.4),
            OneOf([
                RandomContrast(),
                RandomBrightness(),
            ], p=.4),
        ], p=0.9)

    def get_geoometric(self):
        geometric = [
            ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=(-.1, .1), 
                rotate_limit=10, 
                p=.5
            ),
        ]
        return Compose(geometric)

    def get_augmentations(self):
        if self.strength is None:
            return None

        transformations = [
            Compose([
                HorizontalFlip(),
            ], p=1.),
            Compose([
                self.get_photometric(),
                self.get_geoometric(),
            ], p=.95)
        ]
        return Compose(
            transformations,
            bbox_params={
                'format': 'pascal_voc', 
                'min_area': 0., 
                'min_visibility': 0., 
                'label_fields': ['category_id']
            }
        )
