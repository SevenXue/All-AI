from shapely.geometry import LineString

def calculate_distance_with_coords(coord_1, coord_2):
    delta_x = coord_1[0] - coord_2[0]
    delta_y = coord_1[1] - coord_2[1]
    distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
    return distance


def get_cos_with_two_line(line_1, line_2):
    # 计算两个 line 的夹角的 cos 值
    # 如果相交，则计算夹角的 cos； 如果不相交，则计算的是两个夹角中小的
    line_1_p1_coord = list(line_1.coords)[0]
    line_1_p2_coord = list(line_1.coords)[1]

    line_2_p1_coord = list(line_2.coords)[0]
    line_2_p2_coord = list(line_2.coords)[1]

    # 找到线段的交点
    if line_1_p1_coord == line_2_p1_coord or line_1_p1_coord == line_2_p2_coord:
        corner_p_coord = line_1_p1_coord
    else:
        corner_p_coord = line_1_p2_coord

    if corner_p_coord == line_1_p1_coord:
        v1 = [line_1_p2_coord[0] - line_1_p1_coord[0], line_1_p2_coord[1] - line_1_p1_coord[1]]
    else:
        v1 = [line_1_p1_coord[0] - line_1_p2_coord[0], line_1_p1_coord[1] - line_1_p2_coord[1]]

    if corner_p_coord == line_2_p1_coord:
        v2 = [line_2_p2_coord[0] - line_2_p1_coord[0], line_2_p2_coord[1] - line_2_p1_coord[1]]
    else:
        v2 = [line_2_p1_coord[0] - line_2_p2_coord[0], line_2_p1_coord[1] - line_2_p2_coord[1]]

    return cos_vector(v1, v2)


def cos_vector(x, y):
    # 计算2个向量的夹角 cos
    if type(x) == LineString:
        x = (x.coords[1][0] - x.coords[0][0], x.coords[1][1] - x.coords[0][1])
        y = (y.coords[1][0] - y.coords[0][0], y.coords[1][1] - y.coords[0][1])

    if len(x) != len(y):
        print('error input,x and y is not in the same space')
        return None

    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i] * y[i]  # sum(X*Y)
        result2 += x[i] ** 2  # sum(X*X)
        result3 += y[i] ** 2  # sum(Y*Y)
    res = result1 / ((result2 * result3) ** 0.5)

    # 由于计算 cos 的函数的误差问题，会有 cos 数值超出 1 的情况(1.0000000000000002)，此时无法计算其 arccos
    if 0 < res - 1 < 0.0000001:
        res = 1
    elif 0 < -1 - res < 0.0000001:
        res = -1
    return res


def get_line_para(line):
    line_point_coord_list = list(line.coords)
    line.p1 = line_point_coord_list[0]
    line.p2 = line_point_coord_list[1]

    line.a = line.p1[1] - line.p2[1]
    line.b = line.p2[0] - line.p1[0]
    line.c = line.p1[0] * line.p2[1] - line.p2[0] * line.p1[1]


def get_cross_point_coord(l1, l2):
    get_line_para(l1)
    get_line_para(l2)
    d = l1.a * l2.b - l2.a * l1.b

    if d != 0:
        x = (l1.b * l2.c - l2.b * l1.c) * 1.0 / d
        y = (l1.c * l2.a - l2.c * l1.a) * 1.0 / d
        return x, y
    else:
        return None
