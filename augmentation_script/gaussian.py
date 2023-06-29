import numpy as np
import matplotlib.pyplot as plt
import cv2
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


def gaussian_noise(img, mean=0, sigma=0.03):
    # int -> float (標準化)
    img = img / 255
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img.shape)
    # noise + 原圖
    gaussian_out = img + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)

    # 原圖: float -> int (0~1 -> 0~255)

    gaussian_out = np.uint8(gaussian_out * 255)
    return gaussian_out


# file_name = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/27_4.png'
# origin_img = cv2.imread(file_name)
# image = gaussian_noise(origin_img)
# rotated = rotate_img(image)
# save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_gaussian_rotated.png'
# cv2.imwrite(save_path, rotated)

import os
import glob as glob

pillId = os.listdir('/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove/')
# pillId = ['10249']
for pill in pillId:
    total = 0
    count = 25
    folderPath = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/train/pill_sample/{pillId}/{pillId}_*.png'.format(
        pillId=pill)
    for file_name in glob.glob(folderPath):
        total += 1
        origin_img = cv2.imread(file_name)
        image = gaussian_noise(origin_img)
        rotated = rotate_img(image)
        save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/train/pill_gaussian/{pillId}/{original}_{' \
                    'count}.png'.format(pillId=pill, original=pill, count=count)
        # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_sharpen_rotated.png'
        cv2.imwrite(save_path, rotated)
        count += 1
        if total == 5:
            break
