import cv2
from glob import glob
import numpy as np
import re
import os
from geopandas import GeoSeries
from shapely.geometry import Polygon, LineString, MultiPolygon, GeometryCollection, Point
from matplotlib import pyplot as plt
from seq2seq_road import Seq2seq
from scipy.spatial import Voronoi
from shapely.ops import unary_union, polygonize

urls = glob('datasets/partOne/*')

os.makedirs('datasets/circle', exist_ok=True)

ids = []
circle_urls = glob('datasets/circle/*')
for url in circle_urls:
    pattern = re.compile(r'\d+')
    id = pattern.search(url).group()
    ids.append(id)

for url in urls:
    pattern = re.compile(r'\d+')
    id = pattern.search(url).group()
    print(id)
    if id in ids:
        arrangment = {}
        with open('datasets/plans_ordered.txt', 'r') as po:
            lines = po.readlines()
            for line in lines:
                data = eval(line)
                if str(data['plan_id']) == id:
                    print('start')
                    arrangment['id'] = id
                    print(arrangment)
                    arrangment['block'] = data['blocks'][0]['coords']
                    arrangment['buildings'] = []
                    for block in data['blocks']:
                        for building in block['buildings']:
                            arrangment['buildings'].append(building['coords'])
                    break
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

        corners = cv2.goodFeaturesToTrack(gray, 50, 0.13, 10, useHarrisDetector=True)

        paths = []
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(green_img, (x, y), 8, (0, 255, 0), -1)
            # # cv2.imwrite(url.replace('partOne', 'circle'), green_img)
            # cv2.imshow(id, green_img)
            # cv2.waitKey(0) & 0xFF
            # cv2.destroyAllWindows()
            paths.append((x * 0.8536585365853658 - 219.17073170731706, 211.77131782945736 - 0.8410852713178295 * y))

        # arrangment['lines'] = paths

        with open('datasets/re_data.txt', 'a+') as pd:
            pd.write(str(arrangment) + '\n')
#
# with open('datasets/reee_data.txt', 'r') as dpd:
#     paths = dpd.readlines()
#     for path in paths:
#         dataset = eval(path)
#         base_shapes = Polygon(dataset['block'])
#         building_shapes = []
#         for building in dataset['buildings']:
#             building_shapes.append(Polygon(building))
#         road = dataset['lines']
#         road = Seq2seq.sort_paths(road)
#         road.append(road[0])
#         road_shapes = LineString(road)
#
#         # 可视化
#         fig = plt.figure(figsize=(5, 5))
#         ax = fig.add_subplot(111)
#         GeoSeries(base_shapes).plot(ax=ax, color='blue')
#         GeoSeries(building_shapes).plot(ax=ax, color='red')
#         GeoSeries(road_shapes).plot(ax=ax, color='green')
#         GeoSeries(Point((0, 0))).plot(ax=ax, color='yellow')
#         ax.set_aspect(1)
#         plt.axis('off')
#         plt.show()