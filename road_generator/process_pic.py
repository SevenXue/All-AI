import cv2
from glob import glob
import os
from ring_road import *

def adjust_endpoint(x, y, img, kernel=4):
    """
        判断坐标点是否为端点
    :param x: x coordinate
    :param y: y coordinate
    :param img: original img
    :param kernel: radius of neighbourhood
    :return: boolean
    """
    up_img = sum(sum(img[x-kernel:x+kernel, y:y+kernel])) / 255
    down_img = sum(sum(img[x-kernel:x+kernel, y-kernel:y])) / 255
    left_img = sum(sum(img[x-kernel:x, y-kernel: kernel])) / 255
    right_img = sum(sum(img[x:x+kernel, y-kernel:y+kernel])) / 255
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

def reload_pic(url):
    frame = cv2.imread(url)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 提取绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_img = cv2.bitwise_and(frame, frame, mask=mask)

    points = collect_points(green_img)

    city_list = [Point(x=item[0], y=item[1]) for item in points]

    points = genetic_algorithm(population=city_list, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)

    for i in range(len(points)-1):
        cv2.line(green_img, tuple([points[i].x, points[i].y]), tuple([points[i+1].x, points[i+1].y]), (0, 255, 0), 2)
        cv2.imshow('img', green_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.line(green_img, tuple([points[0].x, points[0].y]), tuple([points[-1].x, points[-1].y]), (0, 255, 0), 2)
    cv2.imwrite(url.replace('generator', 'generator_ga'), cv2.add(frame, green_img))
    cv2.imshow('img', green_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

if __name__ == '__main__':
    urls = glob('generator/*')
    os.makedirs('generator_ga', exist_ok=True)
    # for url in urls:
    #     reload_pic(url)
    url = 'generator/40.jpg'
    reload_pic(url)




