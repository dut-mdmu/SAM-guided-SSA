import os
import shutil


def copy_mask(masks_path):
    dirlist = os.listdir(masks_path)
    for d in dirlist:
        source_mask = os.path.join(masks_path, d, 'combined_mask.png')
        target_mask = os.path.join(masks_path, d + '.png')
        shutil.copy(source_mask, target_mask)

masks_path = 'datasets/Suturing_C001_capture1/masks'

copy_mask(masks_path)