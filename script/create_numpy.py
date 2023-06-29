from PIL import Image
from torchvision import transforms
import torch
import glob
import numpy as np
import json
import os
import random


path = "/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/weight/pill_0622_no_noise.pth"
model = torch.load(path)
model.eval()
img_list = []
vector_list = []

try:
    json_file = open('../label/new_pill_class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


def get_vector(im_path):
    img = Image.open(im_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0).to('cuda')

    # Pass input image through feature extractor
    with torch.no_grad():
        feature_map = model.features(img_tensor)  # denseNet
        feature_vector = feature_map.view(-1)  # 將 feature_map 攤平成一個一維向量
        return feature_vector


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def get_label(query_path):
    pill_id = query_path.split('/')[-1].split('_')[0]
    key = get_key_from_value(class_indict, pill_id)
    # pred = class_indict[str(predict_cla)]
    # label = id_to_label[pill_id]
    return key

import copy
def swapPositions(list, pos1, pos2):

    tempList = copy.deepcopy(list)
    tempList[pos1], tempList[pos2] = tempList[pos2], tempList[pos1]
    print(type(list))
    print(type(tempList))
    return tempList

many_res = []
total = 0
count = 0

pill_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train_aug/'
pill_ids = os.listdir(pill_path)
# print(pill_ids)
for pill_id in pill_ids:
    # query_pic_folder = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/train/pill_remove_all/{pill_id}/{pill_id}_*.png'.format(pill_id=pill_id)
    query_pic_folder = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/test_remove_full/{pill_id}_*.png'.format(pill_id=pill_id)
    for query_pic_path in glob.glob(query_pic_folder):
        count += 1
        file = query_pic_path.split('/')[-1]
        label = get_label(query_pic_path)
        fx = get_vector(query_pic_path).cpu().numpy()
        f = []
        query_img = Image.open(query_pic_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(query_img).unsqueeze(0).to('cuda')
        denseNet_result = torch.squeeze(model(img_tensor))
        predict = torch.softmax(denseNet_result, dim=0)
        top5_prob, top5_id = torch.topk(predict, 5)
        top5_id = top5_id.cpu().numpy()
        ids = []
        print(top5_id)
        for id in top5_id:
            ids.append(id)
        # ids.sort()
        # random.shuffle(ids)
        # print(ids, label)
        if int(label) != top5_id[0]:
            print(top5_id)
            print(file)
            total += 1
        # if int(label) not in ids:
        #     print(ids, label)
        for i in range(5):
            pos = i
            if pos < 0:
                pos = 0
            ids = swapPositions(top5_id, 0, pos)
            # print(res)
            if int(label) in ids:
                # random.shuffle(ids)
                print(ids, label, file)
                res = [fx, ids, label, file]
                many_res.append(res)

myRes = np.zeros(len(many_res), dtype=object)
print('total: ', total)
print(count)
print(len(many_res))
for i in range(0, len(many_res)):
    myRes[i] = many_res[i]

np.save('/media/wall/4TB_HDD/0611_finalDBL/numpy/dbl_test_dataset_0623', myRes)
# 22554