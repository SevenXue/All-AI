from matplotlib import pyplot as plt
from math import cos, sin, pi
train_url = 'datasets/train_data_init.txt'
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def spin(angle, point):
    '''
        对点进行旋转30°
    :param point: tuple,point
    :return: tuple, point_30
    '''
    cos_30 = cos(angle / 180 * pi)
    sin_30 = sin(angle / 180 * pi)
    distance = (point[0] ** 2 + point[1] **2) ** 0.5
    cos_a = point[0] / distance
    sin_a = point[1] / distance
    cos_b = cos_a * cos_30 - sin_a * sin_30
    sin_b = sin_a * cos_30 + cos_a * sin_30
    point_30 = (distance * cos_b, distance * sin_b)
    return point_30

with open(train_url, 'r') as td:
    datas = td.readlines()
    blocks = {}
    block_ids = {}
    for data in datas:
        info = eval(data)
        block = str(info['block'])
        id = info['id']
        if block not in blocks.keys():
            blocks[block] = 1
            block_ids[block] = [id]
        else:
            blocks[block] += 1
            block_ids[block].append(id)

    # data statistics
    num = len(blocks.keys())
    max_block = min(11, max(blocks.values()))
    print(f'the num of blocks: {num}')
    print(f"the max of one block's designs: {max_block}")

    # rich data
    for item in blocks.keys():
        if blocks[item] == max_block:
            continue
        else:
            num_block = blocks[item]
            avg = max(1, int((max_block - blocks[item]) / blocks[item]))
            while (blocks[item] < max_block):
                for i in range(num_block):
                    if i == num_block - 1:
                        avg = max_block - blocks[item]
                    if blocks[item] >= max_block:
                        break
                    id_label = block_ids[item][i]
                    for line in datas:
                        infos = eval(line)
                        id = infos['id']
                        if int(id) == int(id_label):
                            for k in range(avg):
                                infos['id'] = str(infos['id']) + alphabet[i]
                                for m in range(len(infos['block'])):
                                    infos['block'][m] = spin(30, infos['block'][m])
                                for building in infos['buildings']:
                                    for n in range(len(building)):
                                        building[n] = spin(30, building[n])
                                for p in range(len(infos['lines'])):
                                    infos['lines'][p] = spin(30, infos['lines'][p])
                                with open(train_url, 'a+') as rd:
                                    rd.write(str(infos) + '\n')
                                blocks[item] += 1
                                if blocks[item] >= max_block:
                                    break
                            break

    # 数据验证
    with open(train_url, 'r') as f:
        line_length = len(f.readlines())
        if line_length >= num * max_block:
            print('rich success!')
        else:
            print(f'the num of dataset is {line_length}, data is not enough!')
