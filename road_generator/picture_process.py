import cv2
import numpy as np
from glob import glob
import os

def adjust_endpoint(x, y, img, kernel=4):
    """
        通过计算邻域内是否有点，判断坐标点是否为端点
    :param x: x coordinate
    :param y: y coordinate
    :param img: 原始图
    :param kernel: 邻域半径
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

def distance(pnta, pntb):
    return ((pnta[0] - pntb[0]) ** 2 + (pnta[1] - pntb[1]) **2) ** 0.5

def line(pnta, pntb):
    k = (pntb[1] - pnta[1]) / (pntb[0] - pnta[0])
    b = pnta[1] - k * pnta[0]
    return k, b

def ring_road(points, img):
    """
        生成环路
    :param points: endpoints
    :param img: 用于画图的图像
    :return: points
    """
    num = len(points)
    if num < 3:
        return []
    sort_points = [points[0]]
    # points排序
    for i in range(1, len(points)):
        label = 0
        for j in range(len(sort_points)):
            if [points[i][0]] < sort_points[j][0]:
                sort_points.insert(j, points[i])
                label = 1
                break
        if label == 0:
            sort_points.append(points[i])

    # 画环路
    start = sort_points[0]
    end = sort_points[-1]
    k, b = line(start, end)
    label_point = sort_points[0]
    upper_points = [sort_points[0]]
    for i in range(1, len(sort_points)):
        if sort_points[i][1] > (k * sort_points[i][0] + b):
            cv2.line(img, label_point, sort_points[i], (0, 255, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            label_point = sort_points[i]
        else:
            upper_points.append(sort_points[i])
    for i in range(1, len(upper_points)):
        cv2.line(img, upper_points[i-1], upper_points[i], (0, 255, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.line(img, upper_points[-1], label_point, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None

    # finished_points = [points[0]]
    # points.remove(finished_points[-1])
    # lines = 0
    #
    # # 搜索最短路径
    # while lines < num - 2:
    #     tmp_item = finished_points[-1]
    #     tmp_point = points[0]
    #     mini = distance(tmp_item, tmp_point)
    #
    #     for item in finished_points:
    #         for point in points:
    #             if distance(item, point) < mini:
    #                 mini = distance(item, point)
    #                 tmp_point = point
    #                 tmp_item = item
    #     #test
    #     cv2.line(img, tmp_item, tmp_point, (0, 255, 0), 2)
    #     lines += 1
    #     if len(finished_points) > 1:
    #         finished_points.remove(tmp_item)
    #     finished_points.append(tmp_point)
    #     points.remove(tmp_point)
    #
    # cv2.line(img, finished_points[0], finished_points[-1], (0, 255, 0), 2)
    # return finished_points

def collect_point(img):
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
    left_point = inital
    right_point = inital
    up_point = inital
    down_point = inital

    for corner in corners:
        x, y = corner.ravel()
        points.append((x, y))
        cv2.circle(img, (x, y), 2, 255, -1)
    #     if adjust_endpoint(y, x, mask, kernel=4):
    #         points.append((x, y))
    #         cv2.circle(img, (x, y), 2, 255, -1)
    #     if x < left_point[0]:
    #         left_point = (x, y)
    #     if x > right_point[0]:
    #         right_point = (x, y)
    #     if y < down_point[1]:
    #         down_point = (x, y)
    #     if y > up_point[1]:
    #         up_point = (x, y)
    # for point in [left_point, right_point, up_point, down_point]:
    #     if point not in points:
    #         points.append(point)
    return points

if __name__ == '__main__':
    urls = glob('generator/*')
    os.makedirs('generator_cv', exist_ok=True)
    # for url in urls:
    while 1:
        url = 'generator/10.jpg'

        frame = cv2.imread(url)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 提取绿色
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([77, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_img = cv2.bitwise_and(frame, frame, mask=mask)

        points = collect_point(green_img)

        ring_road(points, green_img)
        # cv2.imshow('res', green_img)
        # cv2.imwrite(url.replace('generator', 'generator_cv'), cv2.add(frame, green_img))
        # cv2.waitKey(5) & 0xFF
        # cv2.destroyAllWindows()



