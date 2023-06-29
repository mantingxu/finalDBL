import glob as glob
import shutil
import os

pill_ids = os.listdir('/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train/pill_remove/')

for id in pill_ids:
    count = 5
    total = 0
    path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/train/pill_remove_all/{pill_id}/{pill_id}_*.png'.format(pill_id=id)
    for file in glob.glob(path):
        total += 1
        dst = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0621/train/pill_final/{original}/{original}_{' \
              'count}.png'.format(original=id, count=count)
        count += 1
        print(dst)
        shutil.copyfile(file, dst)
        if total == 20:
            break
