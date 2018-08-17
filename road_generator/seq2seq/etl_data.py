import json
from shapely.geometry import Polygon, LineString, Point
from matplotlib import pyplot as plt
from generate import InnerCircleRoadGenerator
from geopandas import GeoSeries


class Building:
    def __init__(self, geom, rotate_angle, floor_height, floor_number=1):
        self.geometry = geom
        self.rotate_angle = rotate_angle
        self.floor_height = floor_height
        self.floor_number = floor_number

def etl(dataset):
    """
        数据清洗
    :param dataset: 原始数据
    :return: 清洗后数据+道路数据
    """
    arrangment = {}
    if len(dataset["blocks"]) > 1:
        return None
    base_shapes = Polygon(dataset["blocks"][0]["coords"])
    arrangment['id'] = dataset['plan_id']
    arrangment['block'] = dataset["blocks"][0]["coords"]
    arrangment['buildings'] = []
    buildings = []
    # building_shapes = []
    for block in dataset["blocks"]:
        for building in block["buildings"]:
            arrangment['buildings'].append(building['coords'])
            building_geom = Polygon(building['coords'])
            building_height = building['height']
            building_obj = Building(geom=building_geom, floor_height=building_height, rotate_angle=0)
            buildings.append(building_obj)
            # building_shapes.append(Polygon(building["coords"]))
    generator = InnerCircleRoadGenerator(base_shapes, buildings=buildings)

    # save data of roads
    if generator.is_fit_to_generate_circle_roads():
        road_line = generator.generate_roads()
        endpoints = []
        for i in range(len(road_line) - 1):
            bound = road_line[i].coords[:]
            for item in bound:
                if item not in endpoints:
                    endpoints.append(item)
        arrangment['lines'] = endpoints

    with open('line_plan.txt', 'a+') as lp:
        lp.write(str(arrangment) + '\n')
    return arrangment, road_line

def test():
    with open('plans.txt', 'r') as plan:
        pl = plan.readlines()
        for line in pl:
            single_data = eval(line.strip('\n'))
            id = single_data['plan_id']
            if id == 94:  # or id == 238 or id == 289 or id == 462 or id == 507:
                data, line = etl(single_data)
                id = data['id']
                base_shapes = data['block']
                buildings = data['buildings']
                # show_picture(id, base_shapes, buildings, line)

if __name__ == '__main__':
    test()

