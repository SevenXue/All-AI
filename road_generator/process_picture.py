import cv2
from glob import glob
import os
from ring_road import *
from scipy.spatial import Voronoi
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection

def adjust_endpoint(x, y, img, kernel=4):
    """
        判断坐标点是否为端点
    :param x: x coordinate
    :param y: y coordinate
    :param img: original img
    :param kernel: radius of neighbourhood
    :return: boolean
    """
    up_img = img[x-kernel:x+kernel, y:y+kernel].sum() / 255
    down_img = img[x-kernel:x+kernel, y-kernel:y].sum() / 255
    left_img = img[x-kernel:x, y-kernel: kernel].sum() / 255
    right_img = img[x:x+kernel, y-kernel:y+kernel].sum() / 255
    tmp = 1
    if up_img > tmp and down_img > tmp:
        return False
    if left_img > tmp and right_img > tmp:
        return False
    else:
        return True

def collect_points(img):
    """
        扫描得到所有的点，用于连接成环路
    :param img:
    :return: Points
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    points = []

    # 初始化边界点
    inital = tuple(corners[0].ravel())

    for corner in corners:
        x, y = corner.ravel()
        points.append((x, y))
    return points

def outline(points, img):
    '''
        生成道路
    :param points: 检测点
    :param img:
    :return: img
    '''
    vor = Voronoi(points)
    lines = []
    for p1, p2 in vor.ridge_points:
        line = LineString([points[p1], points[p2]])
        lines.append(line)

    line_area_polygon = unary_union(list(polygonize(lines)))

    if type(line_area_polygon) == MultiPolygon:
        line_area_polygon = max([geom for geom in line_area_polygon.geoms], key=lambda x: x.area)
        exterior_point_coord_list = list(line_area_polygon.exterior.coords)
    elif type(line_area_polygon) == GeometryCollection and not line_area_polygon.is_empty:
        max_area_collection = max([geom for geom in line_area_polygon], key=lambda x: x.area)
        if type(max_area_collection) == Polygon:
            line_area_polygon = max_area_collection
        elif type(max_area_collection) == MultiPolygon:
            line_area_polygon = max([geom for geom in max_area_collection.geoms], key=lambda x: x.area)

        exterior_point_coord_list = list(line_area_polygon.exterior.coords)
    else:
        exterior_point_coord_list = list(line_area_polygon.exterior.coords)

    for i in range(len(exterior_point_coord_list)-1):
        cv2.line(img, tuple(map(int, exterior_point_coord_list[i])), tuple(map(int, exterior_point_coord_list[i+1])), (0, 255, 0), 2)
    return img

def reload_pic(url):
    '''
        为图片添加道路
    :param url: 图片；链接
    :return:  None
    '''
    frame = cv2.imread(url)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 提取绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_img = cv2.bitwise_and(frame, frame, mask=mask)

    points = collect_points(green_img)
    road_pic = outline(points, green_img)
    # city_list = [Point(x=item[0], y=item[1]) for item in points]
    #
    # points = genetic_algorithm(population=city_list, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)
    cv2.imwrite(url.replace('generator', 'generator_vor'), cv2.add(frame, road_pic))
    return None


if __name__ == '__main__':
    urls = glob('generator/*')
    os.makedirs('generator_vor', exist_ok=True)
    for url in urls:
        reload_pic(url)
    # url = 'images/generator_6/8.jpg'
    # reload_pic(url)