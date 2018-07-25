import json
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from generate import InnerCircleRoadGenerator
from geopandas import GeoSeries
import os

SAVE_DIR = 'datasets/plan'
init = SAVE_DIR + '/init'
building_dir = SAVE_DIR + '/buildings'
os.makedirs(init, exist_ok=True)
os.makedirs(building_dir, exist_ok=True)

class Building:
    def __init__(self, geom, rotate_angle, floor_height, floor_number=1):
        self.geometry = geom
        self.rotate_angle = rotate_angle
        self.floor_height = floor_height
        self.floor_number = floor_number

def show_road(dataset):
    """
        根据数据生成道路图
    :param dataset: 原始数据
    :return: None
    """
    if len(dataset["blocks"]) > 1:
        return None
    base_shapes = Polygon(dataset["blocks"][0]["coords"])
    id = dataset['plan_id']
    print(id)
    buildings = []
    building_shapes = []
    for block in dataset["blocks"]:
        for building in block["buildings"]:
            building_geom = Polygon(building['coords'])
            building_height = building['height']
            building_obj = Building(geom=building_geom, floor_height=building_height, rotate_angle=0)
            buildings.append(building_obj)
            building_shapes.append(Polygon(building["coords"]))
    generator = InnerCircleRoadGenerator(base_shapes, buildings=buildings)

    # save the picture
    if generator.is_fit_to_generate_circle_roads():
        road_line = generator.generate_roads()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        GeoSeries(base_shapes).plot(ax=ax, color='blue')
        GeoSeries(building_shapes).plot(ax=ax, color='red')
        GeoSeries(road_line).plot(ax=ax, color='green', linewidth=4.0)
        plt.axis('off')
        plt.savefig(building_dir + '/' + str(id) + '.jpg')
        plt.close()

def test():
    with open('plans.txt') as plan:
        pl = plan.readlines()
        for line in pl[12259:]:
            single_data = eval(line.strip('\n'))
            show_road(single_data)

if __name__ == '__main__':
    test()

