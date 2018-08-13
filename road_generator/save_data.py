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

def show_road(dataset):
    """
        根据数据生成道路图
    :param dataset: 原始数据
    :return: None
    """
    arrangment = {}
    if len(dataset["blocks"]) > 1:
        return None
    base_shapes = Polygon(dataset["blocks"][0]["coords"])
    arrangment['id'] = dataset['plan_id']
    arrangment['block'] = dataset["blocks"][0]["coords"]
    arrangment['buildings'] = []
    buildings = []
    building_shapes = []
    for block in dataset["blocks"]:
        for building in block["buildings"]:
            building_geom = Polygon(building['coords'])
            arrangment['buildings'].append(building['coords'])
            building_height = building['height']
            building_obj = Building(geom=building_geom, floor_height=building_height, rotate_angle=0)
            buildings.append(building_obj)
            building_shapes.append(Polygon(building["coords"]))
    generator = InnerCircleRoadGenerator(base_shapes, buildings=buildings)

    # save the picture
    if generator.is_fit_to_generate_circle_roads():
        road_line = generator.generate_roads()
        endpoints = []
        for i in range(len(road_line) - 1):
            bound = road_line[i].coords[:]
            for item in bound:
                if item not in endpoints:
                    endpoints.append(item)
        arrangment['lines'] = endpoints
        # reserve_line = []
        # for i in range(len(endpoints)-1):
        #     reserve_line.append(LineString([endpoints[i], endpoints[i+1]]))  #
        #     fig = plt.figure(figsize=(5, 5))
        #     ax = fig.add_subplot(111)
        #     GeoSeries(base_shapes).plot(ax=ax, color='blue')
        #     GeoSeries(building_shapes).plot(ax=ax, color='red')
        #     GeoSeries(reserve_line).plot(ax=ax, color='green', linewidth=4.0)
        #     plt.axis('off')
        #     plt.show()
    return arrangment
def test():
    with open('plans.txt') as plan:
        pl = plan.readlines()
        for line in pl:
            single_data = eval(line.strip('\n'))
            id = single_data['plan_id']
            if id == 94:
                data = show_road(single_data)
                with open('line_plan.txt', 'a+') as lp:
                    lp.write(str(data) + '\n')
                break

if __name__ == '__main__':
    test()

