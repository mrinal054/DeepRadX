import albumentations as A
from albumentations.pytorch import ToTensorV2

def transforms():
    
    transform_list = [
        A.HorizontalFlip(p=0.5),
        
        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=0, shift_limit=0, p=0.5, border_mode=0), # scale only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=10, shift_limit=0, p=0.5, border_mode=0), # rotate only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0), # shift only
                A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0), # affine transform
            ], p=0.7
        ),
        
        A.OneOf(
                [
                    A.Perspective(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.Sharpen(p=0.2),
                    A.Blur(blur_limit=3, p=0.2),
                    A.MotionBlur(blur_limit=3, p=0.2),
                ],
                p=0.5,
            ),
        
        A.ElasticTransform(alpha=3.0, sigma=50.0, alpha_affine=None, p=0.5),
    ]
    
    transform = A.Compose(transform_list, p=0.9) # do augmentation 90% time

    return transform
    
def tta_transforms():
    transform = [
        A.Compose([ToTensorV2()]),  # Original image (no transformation)
        # A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()]),
        # A.Compose([A.VerticalFlip(p=1.0), ToTensorV2()]),
        # A.Compose([A.Rotate(limit=15, p=1.0), ToTensorV2()]),
        # A.Compose([A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0), ToTensorV2()]),
    ]
    
    return transform
    
    