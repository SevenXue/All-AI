import cv2
from glob import glob
import numpy as np
import re
import os
from geopandas import GeoSeries
from shapely.geometry import Polygon, LineString
from matplotlib import pyplot as plt
from seq2seq_road import sort_paths

def generate_pic_of_block(data_name):
    block_ids = get_ids(f'datasets/{data_name}')
    with open('datasets/plans_two.txt', 'r') as pt:
        lines = pt.readlines()
        for line in lines:
            data = eval(line)
            if str(data['plan_id']) in block_ids:
                print(data['plan_id'])
                base_shapes = Polygon(data["blocks"][0]["coords"])
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                GeoSeries(base_shapes).plot(ax=ax, color='blue')
                plt.axis('off')
                ax.set_aspect(1)
                plt.savefig(f'datasets/{data_name}_block/' + data['plan_id'] + '.jpg')
                plt.close()
    return block_ids

def get_opencv_data_of_block(block_urls, data_name):
    if not os.path.exists(block_urls):
        raise ValueError('the path is not right!')

    block_ids = []
    if os.path.exists(f'{block_urls}_opencv'):
        block_ids += get_ids(f'{block_urls}_opencv')

    picture_urls = glob(block_urls + '/*')
    for url in picture_urls:
        pattern = re.compile(r'\d+')
        id = pattern.search(url).group()
        print(id)
        if id not in block_ids:
            build = {}
            with open('datasets/plans_two.txt', 'r') as pt:
                lines = pt.readlines()
                for line in lines:
                    data = eval(line)
                    if str(data['plan_id']) == id:
                        build['id'] = id
                        build['block'] = data['blocks'][0]['coords']
                        build['block'].pop()

                        frame = cv2.imread(url)

                        # 角点检测
                        gary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = np.float32(gary)
                        corners = cv2.goodFeaturesToTrack(gray, 50, 0.15, 10, useHarrisDetector=True)

                        paths = []
                        for corner in corners:
                            x, y = corner.ravel()
                            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                            paths.append((x - 250, 250 - y))

                        if len(paths) == len(build['block']):
                            cv2.imwrite(url.replace(data_name + '_block', data_name + '_block_opencv'), frame)
                            build['track_block'] = paths
                            with open(f'datasets/{data_name}_block.txt', 'a+') as pd:
                                pd.write(str(build) + '\n')
                        break

def get_opencv_data_of_road(block_data, data_name):
    if not os.path.exists(block_data):
        raise ValueError('the path is not right or process block first!')

    block_ids = get_ids(f'datasets/{data_name}_block_opencv')
    road_ids = get_ids(f'datasets/{data_name}_road_opencv')

    with open(block_data, 'r') as bd:
        buildings = bd.readlines()
        for build in buildings:
            build = eval(build)
            build_id = build['id']
            if build_id in block_ids and build_id not in road_ids:
                url = f'datasets/{data_name}/{build_id}.jpg'
                with open('datasets/plans_two.txt', 'r') as pt:
                    lines = pt.readlines()
                    for line in lines:
                        data = eval(line)
                        if str(data['plan_id']) == build_id:
                            build['buildings'] = []
                            for block in data['blocks']:
                                for building in block['buildings']:
                                    build['buildings'].append(building['coords'])

                            frame = cv2.imread(url)
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                            # 提取绿色
                            lower_green = np.array([35, 43, 46])
                            upper_green = np.array([77, 255, 255])
                            mask = cv2.inRange(hsv, lower_green, upper_green)
                            green_img = cv2.bitwise_and(frame, frame, mask=mask)

                            # 角点检测
                            gary = cv2.cvtColor(green_img, cv2.COLOR_BGR2GRAY)
                            gray = np.float32(gary)

                            corners = cv2.goodFeaturesToTrack(gray, 50, 0.15, 10, useHarrisDetector=True)

                            paths = []
                            for corner in corners:
                                x, y = corner.ravel()
                                cv2.circle(green_img, (x, y), 8, (0, 0, 255), -1)
                                paths.append((x - 250, 250 - y))
                            cv2.imwrite(url.replace(data_name, data_name + '_road_opencv'), green_img)
                            build['lines'] = paths

                            with open(f'datasets/{data_name}_tmp.txt', 'a+') as pd:
                                pd.write(str(build) + '\n')
                            break

def choice_and_resize_road_data(road_urls, data_name):

    if not os.path.exists(road_urls):
        raise ValueError('the path is not right or process road first!')

    road_ids = get_ids(road_urls)

    part_ids = []
    if os.path.exists(f'datasets/{data_name}.txt'):
        with open(f'datasets/{data_name}.txt', 'r') as dt:
            datas = dt.readlines()
            for item in datas:
                item = eval(item)
                if item['id'] not in part_ids:
                    part_ids.append(item['id'])

    with open(f'datasets/{data_name}_tmp.txt', 'r') as dt:
        datas = dt.readlines()
        for item in datas:
            item = eval(item)
            if str(item['id']) in road_ids and item['id'] not in part_ids:
                print(item['id'])
                track_block = sort_paths(item['track_block'])
                block = sort_paths(item['block'])
                k_x, b_x, k_y, b_y = calculate_translate_opencv(track_block, block)
                paths = item['lines']
                for i in range(len(paths)):
                    paths[i] = (paths[i][0] * k_x + b_x, paths[i][1] * k_y + b_y)
                arrangment = {}
                arrangment['id'] = item['id']
                arrangment['block'] = item['block']
                arrangment['buildings'] = item['buildings']
                arrangment['lines'] = paths
                with open(f'datasets/{data_name}.txt', 'a+') as dn:
                    dn.write(str(arrangment) + '\n')

    if os.path.exists(f'datasets/{data_name}_tmp.txt'):
        os.remove(f'datasets/{data_name}_tmp.txt')

def calculate_translate_opencv(track_block, block):
    '''
        图片和数据转化
    :param track_block:list,图片检测数据
    :param block: list,采集数据
    :return:
    '''
    k_x = (block[-2][0] - block[0][0]) / (track_block[-2][0] - track_block[0][0])
    print(k_x)
    b_x = block[0][0] - k_x * track_block[0][0]
    print(b_x)

    k_y = (block[-2][1] - block[0][1]) / (track_block[-2][1] - track_block[0][1])
    print(k_y)
    b_y = block[0][1] - k_y * track_block[0][1]
    print(b_y)

    for i in range(len(track_block)):
        track_block[i] = (track_block[i][0] * k_x + b_x, track_block[i][1] * k_y + b_y)

    return k_x, b_x, k_y, b_y

def get_ids(file_url):
    '''
        ids
    :param file_url:str
    :return: list of ids
    '''
    urls = glob(file_url+'/*')
    ids = []
    pattern = re.compile(r'\d+')
    for url in urls:
        id = pattern.search(url).group()
        if id not in ids:
            ids.append(id)
    return ids

def visual_data_of_road(url, save=True):
    '''
        道路数据可视化
    :param url: str, 数据集
    :param save: bool，是否保存图片
    :return: None
    '''
    with open(url, 'r') as dpd:
        paths = dpd.readlines()
        for path in paths:
            dataset = eval(path)
            part_id = dataset['id']
            base_shapes = Polygon(dataset['block'])
            building_shapes = []
            for building in dataset['buildings']:
                building_shapes.append(Polygon(building))
            road = dataset['lines']
            road = sort_paths(road)
            road.append(road[0])
            road_shapes = LineString(road)

            # 可视化
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            GeoSeries(base_shapes).plot(ax=ax, color='blue')
            GeoSeries(building_shapes).plot(ax=ax, color='red')
            GeoSeries(road_shapes).plot(ax=ax, color='green')
            ax.set_aspect(1)
            plt.axis('off')
            if save:
                plt.savefig(f'datasets/part_two_re/{part_id}.jpg')
            else:
                plt.show()
            plt.close()

if __name__ == '__main__':
    part = 'part_two'

    # os.makedirs(f'datasets/{part}_road_opencv', exist_ok=True)
    # os.makedirs(f'datasets/{part}_block_opencv', exist_ok=True)
    # os.makedirs(f'datasets/{part}_block', exist_ok=True)
    os.makedirs(f'datasets/{part}_re', exist_ok=True)
    part_url = f'datasets/{part}'
    road_url = f'datasets/{part}_road'
    block_url = f'datasets/{part}_block'
    block_data = f'datasets/{part}_block.txt'

    # get_opencv_data_of_block(block_url, part)
    # get_opencv_data_of_road(block_data, part)
    # choice_and_resize_road_data(road_url, part)

    road_ids = get_ids('datasets/road_re')
    print(len(road_ids))
    with open('datasets/train_data_not_extraction.txt', 'r') as tdne:
        lines = tdne.readlines()
        print(len(lines))
        for line in lines:
            data = eval(line)
            if str(data['id']) in road_ids:
                road_ids.remove(str(data['id']))
                with open('datasets/train_data_not_extraction_re.txt', 'a+') as tdner:
                    tdner.write(str(data) + '\n')
