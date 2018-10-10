from matplotlib import pyplot as plt
from shapely.geometry import Polygon, LineString
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

with open('../seq2seq/datasets/train_data_extraction.txt', 'r') as dt:
    lines = dt.readlines()
    for line in lines:
        data = eval(line)
        data_id = data['id']
        print(data_id)

        path = Polygon(sort_paths(data['lines']))
        data['type'] =[]
        for i in range(len(data['buildings'])):
            build = Polygon(data['buildings'][i])
            label = 0 if path.distance(build) == 0 else 1
            data['type'].append(label)

        with open('datasets/data_with_type.txt', 'a+') as dwt:
            dwt.write(str(data) + '\n')