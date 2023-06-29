import cv2
import numpy as np
from PIL import Image
import random


def rotate_img(img_cv2):
    img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    # img = Image.open(image_path)

    angle = random.randint(1, 360)
    rotated_img = img.rotate(angle)

    # rotated_img = rotated_img.save(dst_path)
    rotated_img = cv2.cvtColor(np.asarray(rotated_img), cv2.COLOR_RGB2BGR)
    return rotated_img


def modify_contrast_and_brightness(img):
    # 公式： Out_img = alpha*(In_img) + beta
    # alpha: alpha參數 (>0)，表示放大的倍数 (通常介於 0.0 ~ 3.0之間)，能夠反應對比度
    # a>1時，影象對比度被放大， 0<a<1時 影象對比度被縮小。
    # beta:  beta参数，用來調節亮度
    # 常數項 beta 用於調節亮度，b>0 時亮度增強，b<0 時亮度降低。

    array_alpha = np.array([0.8])  # contrast
    array_beta = np.array([1.3])  # brightness

    # add a beta value to every pixel
    img = cv2.add(img, array_beta)

    # multiply every pixel value by alpha
    img = cv2.multiply(img, array_alpha)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255)
    return img


# file_name = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/27_4.png'
# origin_img = cv2.imread(file_name)
# image = modify_contrast_and_brightness(origin_img)
# rotated = rotate_img(image)
# save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_brightness_rotated.png'
# cv2.imwrite(save_path, rotated)

import os
import glob as glob

pillId = os.listdir('/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove/')
# pillId = ['10249']
for pill in pillId:
    count = 0
    folderPath = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_remove/{pillId}_*.png'.format(pillId=pill)
    for file_name in glob.glob(folderPath):
        origin_img = cv2.imread(file_name)
        image = modify_contrast_and_brightness(origin_img)
        rotated = rotate_img(image)
        save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/augmentation/test_brightness/{original}_{' \
                    'count}.png'.format(pillId=pill, original=pill, count=count)
        # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_sharpen_rotated.png'
        cv2.imwrite(save_path, rotated)
        count += 1