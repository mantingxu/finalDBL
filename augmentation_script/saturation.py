import cv2
import numpy as np
from PIL import Image
import random


def modify_lightness_saturation(img):
    origin_img = img

    # 圖像歸一化，且轉換為浮點型
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0

    # 顏色空間轉換 BGR -> HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    lightness = -0.5  # lightness 調整為  "1 +/- 幾 %"
    saturation = 30  # saturation 調整為 "1 +/- 幾 %"

    # 亮度調整
    hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 飽和度調整
    hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 顏色空間反轉換 HLS -> BGR
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = ((result_img * 255).astype(np.uint8))
    return result_img


def rotate_img(img_cv2):
    img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    # img = Image.open(image_path)

    angle = random.randint(1, 360)
    rotated_img = img.rotate(angle)

    # rotated_img = rotated_img.save(dst_path)
    rotated_img = cv2.cvtColor(np.asarray(rotated_img), cv2.COLOR_RGB2BGR)
    return rotated_img


# file_name = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/27_4.png'
# origin_img = cv2.imread(file_name)
# image = modify_lightness_saturation(origin_img)
# rotated = rotate_img(image)
# save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_saturation_rotated.png'
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
        image = modify_lightness_saturation(origin_img)
        rotated = rotate_img(image)
        save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/augmentation/test_saturation/{original}_{' \
                    'count}.png'.format(pillId=pill, original=pill, count=count)
        # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/test/test_sharpen_rotated.png'
        cv2.imwrite(save_path, rotated)
        count += 1