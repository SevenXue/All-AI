from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from generate import InnerCircleRoadGenerator
from geopandas import GeoSeries
import os
from glob import glob
import re

SAVE_DIR = 'datasets/'
init_dir = SAVE_DIR + 'init'
building_dir = SAVE_DIR + 'ai_road'
os.makedirs(init_dir, exist_ok=True)
os.makedirs(building_dir, exist_ok=True)

class Building:
    def __init__(self, geom, rotate_angle, floor_height, floor_number=1):
        self.geometry = geom
        self.rotate_angle = rotate_angle
        self.floor_height = floor_height
        self.floor_number = floor_number

def create_data_of_ring_road(dataset):
    """
        根据数据生成道路图
    :param dataset: 原始数据
    :return: None
    """

    arrangment = {}
    if len(dataset["blocks"]) > 1:
        return None
    base_shapes = Polygon(dataset["blocks"][0]["coords"])
    id = dataset['plan_id']
    arrangment['id'] = id
    arrangment['block'] = dataset['blocks'][0]['coords']
    arrangment['buildings'] = []
    buildings = []
    building_shapes = []
    for block in dataset["blocks"]:
        for building in block["buildings"]:
            arrangment['buildings'].append(building['coords'])
            building_geom = Polygon(building['coords'])
            building_height = building['height']
            building_obj = Building(geom=building_geom, floor_height=building_height, rotate_angle=0)
            buildings.append(building_obj)
            building_shapes.append(Polygon(building["coords"]))
    generator = InnerCircleRoadGenerator(base_shapes, buildings=buildings)

    if generator.is_fit_to_generate_circle_roads():
        road_line = generator.generate_roads()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        GeoSeries(base_shapes).plot(ax=ax, color='blue')
        GeoSeries(building_shapes).plot(ax=ax, color='red')
        GeoSeries(road_line).plot(ax=ax, color='green', linewidth=4.0)
        plt.axis('off')
        ax.set_aspect(1)
        plt.savefig(building_dir + '/' + str(id) + '.jpg')
        plt.close()

        # save datasets
        endpoints = []
        for i in range(len(road_line) - 1):
            bound = road_line[i].coords[:]
            for item in bound:
                if item not in endpoints:
                    endpoints.append(item)
        arrangment['lines'] = endpoints

        with open('datasets/select_plan.txt', 'a+') as lp:
            lp.write(str(arrangment) + '\n')

def filter_data_by_generate_road(url):
    '''
        通过道路生成筛选数据
    :return:
    '''
    ids = []
    with open('datasets/plans.txt', 'r') as pl:
        datas = pl.readlines()
        for data in datas:
            data = eval(data)
            if data['plan_id'] not in ids:
                ids.append(data['plan_id'])
    with open('datasets/plans_ordered.txt', 'r') as po:
        datas = po.readlines()
        for data in datas:
            data = eval(data)
            if data['plan_id'] not in ids:
                ids.append(data['plan_id'])


    with open(url, 'r') as plan:
        pl = plan.readlines()
        for i in range(len(pl[:])):
            print(i)
            single_data = eval(pl[i].strip('\n'))
            if single_data['plan_id'] not in ids:
                create_data_of_ring_road(single_data)

def filter_data_by_choose_picture(url):
    '''
        人工筛选图片，挑选数据
    :param url:str
    :return:
    '''
    datasets = glob(url)
    ids = []
    for data in datasets:
        pattern = re.compile(r'\d+')
        id = pattern.search(data).group()
        if id not in ids:
            ids.append(id)

    with open('datasets/select_plan.txt', 'r') as lp:
        paths = lp.readlines()
        for path in paths:
            path = eval(path)
            if str(path['id']) in ids:
                ids.remove(str(path['id']))
                print(path['id'])
                with open('datasets/train_data.txt', 'a+') as td:
                    td.write(str(path) + '\n')

    with open('train_data.txt', 'r') as td:
        print(len(td.readlines()))

if __name__ == '__main__':
    # filter_data_by_choose_picture('datasets/xd_road/*')
    filter_data_by_generate_road('datasets/ai_data.txt')