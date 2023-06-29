import cv2
from PIL import Image
import random
import numpy as np
import os
import glob as glob


def sharpen(img, sigma=10):
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm


def rotate_img(img_cv2):
    img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    # img = Image.open(image_path)

    angle = random.randint(1, 360)
    rotated_img = img.rotate(angle)

    # rotated_img = rotated_img.save(dst_path)
    rotated_img = cv2.cvtColor(np.asarray(rotated_img), cv2.COLOR_RGB2BGR)
    return rotated_img


pillId = os.listdir('/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove/')
# pillId = ['10249']
for pill in pillId:
    count = 0
    folderPath = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_remove/{pillId}_*.png'.format(pillId=pill)
    for file_name in glob.glob(folderPath):
        origin_img = cv2.imread(file_name)
        image = sharpen(origin_img)
        rotated = rotate_img(image)
        save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/augmentation/test_sharpen/{original}_{' \
                    'count}.png'.format(pillId=pill, original=pill, count=count)
        # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_sharpen_rotated.png'
        cv2.imwrite(save_path, rotated)
        count += 1
