import glob
import os
import random
import cv2
from PIL import Image


def add_noise(img):
    # img = cv2.imread(img_path)
    # Getting the dimensions of the image
    row, col, ch = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(30, 50)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(30, 50)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


import numpy


def rotate_img(image_path, dst_path):
    img = Image.open(image_path)

    angle = random.randint(1, 360)
    rotated_img = img.rotate(angle)

    # rotated_img = rotated_img.save(dst_path)
    rotated_img = cv2.cvtColor(numpy.asarray(rotated_img), cv2.COLOR_RGB2BGR)
    return rotated_img


# salt-and-pepper noise can
# be applied only to grayscale images
# Reading the color image in grayscale image
# image_path = './dataset/test/12083/12083_0-0.png'
# pillId = ['12448', '12222', '12083', '325', '2311', '2321', '4061', '4115', '6356']
pillId = os.listdir('/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove/')
# pillId = ['10249']
for pill in pillId:
    count = 0
    folderPath = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_remove/{pillId}_*.png'.format(pillId=pill)
    for file in glob.glob(folderPath):
        print(file)
        img = cv2.imread(file)
        name = file.split('/')[-1].replace('.png', '')
        original = name.split('_')[0]
        # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove_aug/{pillId}/{original}_{' \
        #             'count}.png'.format(pillId=original, original=original, count=count)
        # print('original:', count)
        # cv2.imwrite(save_path, img)  # 0
        # count += 1
        #
        for i in range(1):
            # print('aug:', count)
            save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/{original}_{' \
                        'count}.png'.format(pillId=original, original=original, count=count)
            dst = rotate_img(file, save_path)
            # count += 1
            # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove_aug/{pillId}/{original}_{' \
            #             'count}.png'.format(pillId=original, original=original, count=count)
            # save_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/test_append/{original}_{' \
            #             'count}.png'.format(pillId=original, original=original, count=count)
            cv2.imwrite(save_path, add_noise(dst))  # 1
            count += 1
