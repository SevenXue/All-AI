from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString
import math
import numpy as np
from matplotlib import pyplot as plt
from geopandas import GeoSeries

def sort_paths(endpoints):
    '''
        对点进行逆时针排序
    :param endpoints: list of points
    :return: sorted points
    '''

    # 4象限
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    for i in range(len(endpoints)):
        if endpoints[i][0] > 0 and endpoints[i][1] > 0:
            x1.append(endpoints[i])
        elif endpoints[i][0] < 0 and endpoints[i][1] > 0:
            x2.append(endpoints[i])
        elif endpoints[i][0] < 0 and endpoints[i][1] < 0:
            x3.append(endpoints[i])
        elif endpoints[i][0] > 0 and endpoints[i][1] < 0:
            x4.append(endpoints[i])

    x1.sort(key=lambda x: (x[1] / x[0]))
    x2.sort(key=lambda x: (x[1] / x[0]))
    x3.sort(key=lambda x: (x[1] / x[0]))
    x4.sort(key=lambda x: (x[1] / x[0]))

    points = x1 + x2 + x3 + x4
    return points

def add_label_of_data(data_file):
    """
        Add label of building
    :param data_file: str url of init data
    :return: None
    """
    with open(data_file, 'r') as dt:
        lines = dt.readlines()
        for line in lines:
            data = eval(line)
            data_id = data['id']
            print(data_id)

            path = Polygon(sort_paths(data['lines']))
            data['type'] = []
            for i in range(len(data['buildings'])):
                build = Polygon(data['buildings'][i])
                label = 1 if path.distance(build) == 0 else 0
                data['type'].append(label)

            with open('datasets/data_with_type.txt', 'a+') as dwt:
                dwt.write(str(data) + '\n')

def base_info_of_block(block):
    """
        Get base information of a block
    :param block:list, end-points of block
    :return:int*4
    """
    if isinstance(block, Polygon):
        x_min, y_min, x_max, y_max = block.bounds
    else:
        x0, y0 = block[0]
        x_min, x_max = x0, x0
        y_min, y_max = y0, y0
        for x, y in block[1:]:
            x_min = min(x, x_min)
            x_max = max(x, x_max)
            y_min = min(y, y_min)
            y_max = max(y, y_max)
    return x_min, x_max, y_min, y_max

def create_img_of_data(base, x_min, x_max, y_min, y_max):
    """
        Create image contains by data
    :param base: list, includes points of data
    :param x_min: int
    :param x_max: int
    :param y_min: int
    :param y_max: int
    :return: img , object of Image
    """
    x_span = x_max - x_min
    y_span = y_max - y_min
    span = math.ceil(max(x_span, y_span))
    img = Image.new("L", (span, span), 0)

    x_sup = (span - x_span) / 2
    y_sup = (span - y_span) / 2
    if len(base[0]) == 2:
        base = [base]
    for item in base:
        coords = [(x - x_min + x_sup, y_max - y + y_sup) for x, y in item]
        ImageDraw.Draw(img).polygon(coords, outline=1, fill=1)

    return img

def reshape_img(img, dim):
    """
        Transform img to ndarray
    :param img: object, img
    :param dim: int, the dimension of ndarray
    :return: ndarray
    """
    img = img.resize((dim, dim))
    shape = np.array(img).reshape((1, dim, dim))
    return shape

def distance_from_building_to_block(block, building):
    """
        Calculate the min distance between block and building
    :param block: list, points of block
    :param building: list, points of building
    :return: int, min distance
    """
    if block[-1] != block[0]:
        block.append(block[0])
    if building[-1] != building[0]:
        building.append(building[0])
    block = LineString(block)
    building = LineString(building)
    return block.distance(building)

def max_min_standard(list):
    """

    :param list:
    :return:
    """
    max_num = max(list)
    min_num = min(list)
    return [(list[i] - min_num) / (max_num - min_num) for i in range(len(list))]

def save_date(data, npz_file):
    """
        Save shapes, styles, distances into a npz file
    :param data: dict, data including shapes, styles, distances
        shapes: (n, dim, dim) ndarray, shapes of blocks
        styles: (n, 1) ndarray, label of n buildings, includes 0 and 1
        distances: (n, 1) ndarray, distance between building and blocks
    :param npz_file: str, name of file that stores 3 ndarray
    """
    shapes = data['shapes']
    styles = data['styles']
    distances = data['distances']
    np.savez(npz_file, shapes=shapes, styles=styles, distances=distances)

def prepare_data(data_file, npz_file=None):
    """
        Prepare data of classify model
    :param data_file: str, name of file that stores data of blocks, buildings, paths and labels
    :param npz_file: str name of file to save data
    :return: dict, data including block, buildings, labels.
    """

    with open(data_file, 'r') as dws:
        lines = dws.readlines()
        shapes = []
        styles = []
        distances = []
        for line in lines:
            data = eval(line)
            block = data['block']
            buildings = data['buildings']
            data_type = data['type']
            x_min, x_max, y_min, y_max = base_info_of_block(block)
            standard_distance = min(y_max-y_min, x_max-x_min)
            block_img = create_img_of_data(block, x_min, x_max, y_min, y_max)
            block_data = reshape_img(block_img, 256)
            building_distances = []
            for i in range(len(buildings)):
                # fig = plt.figure(figsize=(5, 5))
                # ax = fig.add_subplot(111)
                # ax.set_aspect(1)
                # GeoSeries(LineString(block)).plot(ax=ax, color='blue')
                # GeoSeries(LineString(buildings[i])).plot(ax=ax, color='red')
                # plt.show()
                # distance = 1 - distance_from_building_to_block(block, buildings[i]) / standard_distance
                # distances.append(distance)
                distance = distance_from_building_to_block(block, buildings[i])
                building_distances.append(distance)
                building_img = create_img_of_data(buildings[i], x_min, x_max, y_min, y_max)
                data_building = np.concatenate((block_data, reshape_img(building_img, 256)))
                buildings_img = create_img_of_data(buildings[:i] + buildings[i+1:len(buildings)], x_min, x_max, y_min, y_max)
                data_buildings = np.concatenate((data_building, reshape_img(buildings_img, 256)))
                shapes.append(data_buildings)
            styles += data_type
            distances += max_min_standard(building_distances)
        print(len(distances))
        print(distances[:20])
        print(max(distances))
        print(min(distances))
        shapes = np.array(shapes)
        n = len(styles)
        styles = np.array(styles).reshape((n, 1))
        distances = np.array(distances).reshape((n, 1))
        data = {
            'shapes': shapes,
            'styles': styles,
            'distances': distances
        }
        if npz_file is not None:
            save_date(data, npz_file)
        return data

if __name__ == '__main__':
    data_file = 'datasets/test_data.txt'
    add_label_of_data(data_file)
    prepare_data('datasets/data_with_type.txt', 'datasets/test_data.npz')