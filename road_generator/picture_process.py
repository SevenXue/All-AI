import cv2
import numpy as np

def adjust_endpoint(x, y, img, kernel=4):
    """
        通过计算邻域内是否有点，判断坐标点是否为端点
    :param x:
    :param y:
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

def ring_road(points, img):
    """
        dijkstra算法，生成环路
    :param points: endpoints
    :param img: 用于画图的图像
    :return: points
    """
    num = len(points)
    finished_points = [points[0]]
    points.remove(finished_points[-1])
    while len(finished_points) < num:
        item = finished_points[-1]
        tmp_point = points[0]
        mini = distance(item, tmp_point)
        #for item in finished_points:

        for point in points:
            if distance(item, point) < mini:
                mini = distance(item, point)
                tmp_point = point
                #tmp_item = item

        cv2.line(img, item, tmp_point, (0, 255, 0), 2)
        finished_points.append(tmp_point)
        points.remove(tmp_point)

    cv2.line(img, finished_points[0], finished_points[-1], (0, 255, 0), 2)
    return finished_points

def collect_point(img):
    """
        集成所有端点
    :param img:
    :return: Points
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    points = []

    # 初始化边界点
    inital = (corners[0].ravel())
    left_point = inital
    right_point = inital
    up_point = inital
    down_point = inital

    for corner in corners:
        x, y = corner.ravel()
        if adjust_endpoint(y, x, mask, kernel=4):
            points.append((x, y))
        if x < left_point[0]:
            left_point = (x, y)
        if x > right_point[0]:
            right_point = (x, y)
        if y < down_point[1]:
            down_point = (x, y)
        if y > up_point[1]:
            up_point = (x, y)
    for point in [left_point, right_point, up_point, down_point]:
        if point not in points:
            points.append(point)
    print(left_point, right_point, up_point, down_point)
    print(points)
    return points

if __name__ == '__main__':
    frame = cv2.imread('generator.jpg')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    points = collect_point(res)

    points = ring_road(points, res)

    cv2.imshow('res', res)

    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()



