import cv2
from PIL import Image
import random
import numpy as np
import os
import glob as glob


def rotate_img(img_cv2, dst_path):
    img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    # img = Image.open(image_path)

    angle = random.randint(1, 360)
    rotated_img = img.rotate(angle)

    rotated_img = rotated_img.save(dst_path)
    # rotated_img = cv2.cvtColor(np.asarray(rotated_img), cv2.COLOR_RGB2BGR)
    return rotated_img


# pillId = os.listdir('/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove/')
pillId = ['12353']
for pill in pillId:
    count = 40
    folderPath = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/train/pill_remove_all/{pillId}/{pillId}_*.png'.format(pillId=pill)
    for i in range(1):
        for file_name in glob.glob(folderPath):
            origin_img = cv2.imread(file_name)
            save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/train/pill_append/{original}_{' \
                        'count}.png'.format(original=pill, count=count)
            # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_sharpen_rotated.png'
            rotated = rotate_img(origin_img, save_path)
            count += 1

