import random

import copy


def swapPositions(list, pos1, pos2):
    tempList = copy.deepcopy(list)
    tempList[pos1], tempList[pos2] = tempList[pos2], tempList[pos1]
    return tempList


arr = [43, 114, 154, 157, 67]
# arr1 = array([43, 114, 154, 157, 67])
for i in range(0, len(arr)):
    pos = i
    if pos < 0:
        pos = 0
    res = swapPositions(arr, 0, pos)
    print(res)
    # print('arr', arr)
