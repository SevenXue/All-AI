#！道路生成
import math
import random

from scipy.spatial import Voronoi
from shapely.affinity import rotate
from shapely.ops import polygonize, unary_union, nearest_points, cascaded_union
from shapely.prepared import prep
from shapely.geometry import Point, MultiPolygon, LineString, Polygon, GeometryCollection
import logging

import util

logger = logging.getLogger()


class InnerCircleRoadGenerator:
    def __init__(self, base, buildings, commercial_buildings=[], office_buildings=[]):
        self.base = base.simplify(0.01)  # 基地 geom
        self.buildings = buildings  # 住宅建筑(道路覆盖需要考虑的部分)
        self.commercial_buildings = commercial_buildings  # 商业建筑,道路需要绕开的部分
        self.office_buildings = office_buildings  # 办公建筑,道路需要绕开的部分

        self.road_type = random.choice([0, 1])  # 道路算法的 2 个不同选点逻辑,随机选择
        self.road_width = 4  # 道路宽度
        self.road_buffer_distance = 20  # 生成环线之后,通过 buffer + parallel 进行平滑处理时的参数
        self.random_threshold_road_cover = random.choice([0.9, 0.95])  # 道路微调过程中,衡量道路对建筑覆盖率变化的参数
        self.min_distance_limit = random.choice([10, 15, 20])  # 距离建筑多远算覆盖建筑
        self.building_buffer_distance = random.choice([4, 6, 8])

    def plan_to_layout(self, height_threshold=25):
        building_box_geom_list = []
        obstacle_box_geom_list = []

        building_geom_list = []
        obstacle_building_list = self.commercial_buildings + self.office_buildings
        obstacle_geom_list = [building.geometry for building in obstacle_building_list]

        for building in self.buildings:
            angle = building.rotate_angle
            building_geom_without_rotate = rotate(building.geometry, -angle)
            building_box_geom_without_rotate = building_geom_without_rotate.envelope
            building_height = building.floor_height * building.floor_number

            building_box_geom = self.get_building_bounding_box(base_geometry=self.base,
                                                               building_geometry=building_box_geom_without_rotate,
                                                               angle=angle)

            if building_height > height_threshold:
                building_box_geom_list.append(building_box_geom)
                building_geom_list.append(building.geometry)
            else:
                obstacle_box_geom_list.append(building_box_geom)
                obstacle_geom_list.append(building.geometry)

        layout = (building_box_geom_list, obstacle_box_geom_list,
                  building_geom_list, obstacle_geom_list)
        return layout

    def find_valid_area(self, base_geom, building_geom_list, obstacle_geom_list):
        """
        用于得到道路可建部分的,并不去除其中的 obstacle
        :param base_geom:
        :param building_geom_list:
        :param obstacle_geom_list:
        :return:
        """
        logger.info('计算基地内道路可分布面积')
        base_polygon = Polygon(base_geom)

        if self.road_type == 0:
            logger.info('使用第一种取点方式生成道路')
            # 如果建筑小于 4 栋，直接取其正常的凸包作为 valid area
            if len(building_geom_list) < 4:
                valid_area = MultiPolygon(building_geom_list).convex_hull.intersection(base_geom)

            # 如果建筑大于 4 栋，但是总面积在 4w 之内，相当于平均边长在 200 米之内，为了增加自由度，将 valid 增大到 距离边界 5 m 的位置
            elif base_polygon.area < 40000:
                valid_area_1 = MultiPolygon(building_geom_list).convex_hull.intersection(base_geom)
                valid_area_2 = Polygon(base_geom).buffer(-5)
                valid_area = valid_area_1.union(valid_area_2)

            # 如果建筑物大于 8 栋，但是总面积在 10w 之内
            elif len(building_geom_list) > 8 and base_polygon.area < 100000:
                valid_area = Polygon(base_geom).buffer(-10)
            else:
                valid_area_1 = MultiPolygon(building_geom_list).convex_hull.intersection(base_geom)
                valid_area_2 = Polygon(base_geom).buffer(-20)
                valid_area = valid_area_1.union(valid_area_2)

            # res = valid_area.difference(valid_obstacle) if valid_obstacle else valid_area
            res = valid_area
            return res
        else:
            logger.info('使用第二种取点方式生成道路')
            valid_area = MultiPolygon(building_geom_list).convex_hull.intersection(base_geom).buffer(-3)
            res = valid_area
            return res

    def find_and_connect_marginal_points(self, points, valid_area, building_geom_list, obstacle_geom_list, distance=4):
        # 通过选择候选点，作为路网可能经过的点
        point_coord_list = [point.coords[0] for point in points]
        distance -= 1

        # 建筑轮廓生成多个多边形
        multi_buffered_building = MultiPolygon([geom.buffer(self.road_width / 2) for geom in building_geom_list])
        if not multi_buffered_building.is_valid:
            multi_buffered_building = multi_buffered_building.buffer(0)

        # 路网覆盖建筑之外的不可建区域
        buffered_obstacle_geom = cascaded_union([geom.buffer(self.road_width / 2) for geom in obstacle_geom_list])

        # 维罗尼多边形
        vor = Voronoi(point_coord_list)
        lines = []
        for p1, p2 in vor.ridge_points:
            line = LineString([point_coord_list[p1], point_coord_list[p2]])
            if (line.within(valid_area)
                    and not line.intersects(multi_buffered_building)
                    and not line.intersects(buffered_obstacle_geom)):
                lines.append(line)

        # 保留外轮廓的线段
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

        lines = []
        for idx in range(len(exterior_point_coord_list) - 1):
            line = LineString([exterior_point_coord_list[idx], exterior_point_coord_list[idx + 1]])
            lines.append(line)

        # 刪除各种不符合规范的线(获得的是变化过程中各个版本的中间情况的 lines )
        final_lines_list = self.drop_error_line(lines, valid_area, multi_buffered_building, buffered_obstacle_geom)
        return final_lines_list

    def building_geom_list_to_point_geom_list(self, building_geom_list, valid_area, obstacle_geom_list):
        # 住宅建筑和障碍物(主要是低层住宅和商业等不可建区域)
        total_building_obstacle_geom = cascaded_union([geom.buffer(self.road_width / 2)
                                                       for geom in (building_geom_list + obstacle_geom_list)])

        # 建筑向外进行 buffer, 得到 box 的四个顶点
        buffered_buildings = [building.buffer(self.building_buffer_distance, join_style=2)
                              for building in building_geom_list]

        # 按照建筑朝向取得交叉点
        road_candidate_point_coord_list = self.get_cross_point(buffered_buildings)

        # 去除靠近 valid area 过近的点,尽量选择内环线，同时保证道路不会距离基地边界过近
        num_building = len(buffered_buildings)
        if num_building > 3:
            road_candidate_point_coord_list = self.drop_point_near_valid_area_border(road_candidate_point_coord_list,
                                                                                     valid_area,
                                                                                     distance=10)
        else:
            road_candidate_point_coord_list = self.drop_point_near_valid_area_border(road_candidate_point_coord_list,
                                                                                     valid_area,
                                                                                     distance=6)

        points = [Point(coord) for coord in road_candidate_point_coord_list
                  if Point(coord).within(valid_area) and not Point(coord).within(total_building_obstacle_geom)]
        return points

    @staticmethod
    def drop_point_near_valid_area_border(candidate_point_coord_list, valid_area, distance=15):
        # 去除靠近 valid area 过近的点,尽量选择内环线，同时保证道路不会距离基地边界过近

        # 收集 valid_area 的边界轮廓线段
        point_coord_list = list(valid_area.exterior.coords)
        border_lines = []
        for idx in range(len(point_coord_list) - 1):
            line = LineString((point_coord_list[idx], point_coord_list[idx + 1]))
            border_lines.append(line)

        # 计算各个点到 有效面积边界 的最短距离
        point_coord_with_min_distance = []
        for coord in candidate_point_coord_list:
            point = Point(coord)
            coord_with_min_distance = [coord, point.distance(min(border_lines, key=lambda x: point.distance(x)))]
            point_coord_with_min_distance.append(coord_with_min_distance)

        # 对各个点到有效面积边界距离进行排序
        point_coord_with_min_distance_in_order = sorted(point_coord_with_min_distance,
                                                        key=lambda coord_with_dis: coord_with_dis[1])

        # 如果所有点都过近,则不进行删减：
        if point_coord_with_min_distance_in_order[-1][1] < distance:
            return [coord_with_dis[0] for coord_with_dis in point_coord_with_min_distance_in_order]
        else:
            return [coord_with_dis[0] for coord_with_dis in point_coord_with_min_distance_in_order if
                    coord_with_dis[1] >= distance]

    @staticmethod
    def get_cross_point(buffered_buildings):
        # 记录建筑box周边四个顶点，这部分点是道路组成的基础，优先保留
        building_box_point_coord_list = []
        for buffered_building in buffered_buildings:
            coord_list = buffered_building.boundary.coords
            building_box_point_coord_list.extend(coord_list[:-1])

        # 生成各个建筑周边 bounding box 的线段
        line_list = []
        for buffered_building in buffered_buildings:
            coord_list = buffered_building.boundary.coords
            for idx in range(len(coord_list) - 1):
                line = LineString([coord_list[idx], coord_list[idx + 1]])
                line_list.append(line)

        # 计算以上各个线段所在直线的交点：
        cross_point_coord_list = []
        for idx_1 in range(len(line_list) - 1):
            line_1 = line_list[idx_1]
            for idx_2 in range(idx_1 + 1, len(line_list)):
                line_2 = line_list[idx_2]
                point_coord = util.get_cross_point_coord(line_1, line_2)
                if point_coord:
                    cross_point_coord_list.append(point_coord)

        # 初步去重
        cross_point_coord_list = list(set(cross_point_coord_list))

        # 去掉间距小于 threshold 的点
        threshold = 4

        # 优先保留建筑周围的轮廓的顶点：
        cross_point_no_near_coord_list = []
        for cross_point_coord in cross_point_coord_list:
            # 计算以上交点到建筑box顶点的距离的最小值，如果距离其中任意顶点过近，则不保留该点
            min_distance_box_point_coord = min(building_box_point_coord_list,
                                               key=lambda x: util.calculate_distance_with_coords(x, cross_point_coord))
            min_distance = util.calculate_distance_with_coords(min_distance_box_point_coord, cross_point_coord)
            if min_distance > threshold:
                cross_point_no_near_coord_list.append(cross_point_coord)

        # 对 cross point 中相近的点进行取舍
        first_index = 0
        while first_index < len(cross_point_no_near_coord_list):
            first_point_coord = cross_point_no_near_coord_list[first_index]

            second_index = first_index + 1
            while second_index < len(cross_point_no_near_coord_list):
                second_point_coord = cross_point_no_near_coord_list[second_index]
                distance = util.calculate_distance_with_coords(first_point_coord, second_point_coord)

                if distance < threshold:
                    # 如果移除当前点，则 second_index 位置的点自动更新为下一个
                    cross_point_no_near_coord_list.remove(second_point_coord)
                else:
                    # 否则需要增加 index
                    second_index += 1
            first_index += 1
        # 最终保留的点为两类： 各线段所在直线交点 + 建筑 box 顶点
        return cross_point_no_near_coord_list + building_box_point_coord_list

    @staticmethod
    def get_building_bounding_box(base_geometry, building_geometry, angle):
        # 对建筑周边box进行提取，并非简单提取 bounding box，而是考虑和建筑最近的基地边界的走向，沿该方向提取建筑 box
        base_point_coord_list = list(base_geometry.exterior.coords)
        building_corner_point = Point([building_geometry.centroid.x, building_geometry.centroid.y])

        # 获取当前基地轮廓线的各个边及其边到当前建筑的距离
        base_border_line_with_building_distance_list = []
        for idx in range(len(base_point_coord_list) - 1):
            base_border_line = LineString([base_point_coord_list[idx], base_point_coord_list[idx + 1]])
            if base_border_line.length > 0.1:
                tmp_distance = building_corner_point.distance(base_border_line)
                base_border_line_with_building_distance_list.append([base_border_line, tmp_distance])

        # 将基地的各个边按照到当前建筑的距离进行排序
        base_border_line_with_building_distance_list = sorted(base_border_line_with_building_distance_list,
                                                              key=lambda border_with_distance: border_with_distance[1])

        # 距离最近的基地周边
        min_line = base_border_line_with_building_distance_list[0][0]
        building_base_min_distance = base_border_line_with_building_distance_list[0][1]

        # 计算建筑各个边和最近的基地周边的夹角，找到最小旋转角度
        building_point_coord_list = list(building_geometry.exterior.coords)
        building_line_list = []
        for idx in range(len(building_point_coord_list) - 1):
            base_border_line = LineString([building_point_coord_list[idx], building_point_coord_list[idx + 1]])
            building_line_list.append(base_border_line)

        # 找到建筑物的最短便和最长边
        longest_line = max(building_line_list, key=lambda x: x.length)
        shortest_line = min(building_line_list, key=lambda x: x.length)

        # 如果该建筑的长宽比小于3(某些长宽比特别大的建筑，不需要适应基地边界方向), 同时该建筑到最近基地边界的距离超过 30
        if longest_line.length / shortest_line.length < 3:
            if building_base_min_distance < 30:
                cos_list_with_building_border = []
                # 计算该基地边和建筑box四个边的夹角的余弦
                for idx in range(4):
                    building_border_line = LineString([list(building_geometry.exterior.coords)[idx],
                                                       list(building_geometry.exterior.coords)[idx + 1]])
                    tmp_cos = util.cos_vector(building_border_line, min_line)
                    cos_list_with_building_border.append([tmp_cos, building_border_line])

                # 找到最小旋转角度的 cos 值以及对应的建筑 border line
                max_cos_with_border = max(cos_list_with_building_border,
                                          key=lambda cos_and_border_line: cos_and_border_line[0])

                # 由于边界和建筑的相对位置不确定，所以此时计算的最小旋转角度可能是顺时针也可能是逆时针
                candidate_building_base_angle_list = [math.acos(max_cos_with_border[0]) / math.pi * 180,
                                                      -math.acos(max_cos_with_border[0]) / math.pi * 180]

                # 将最近的基地边按照其中一个角度旋转，计算其和建筑中该边的夹角，以确定建筑旋转到和该边界平行所需的具体角度
                rotated_border_line = rotate(max_cos_with_border[1], candidate_building_base_angle_list[0],
                                             origin='centroid')
                tmp_cos = util.cos_vector(rotated_border_line, min_line)

                if 1 - 0.0001 < tmp_cos < 1 + 0.0001:
                    rotate_angle = candidate_building_base_angle_list[0]
                else:
                    rotate_angle = candidate_building_base_angle_list[1]

                # 由于建筑本身的旋转角度只能是逆时针，会出现很多大于180 度的旋转角，但是这些角度在计算 envelope 的旋转的时候，是多余的
                fit_angle = abs(180 - angle)

                # 先建筑反向旋转，然后通过 envelope 函数产生正南北方向的外包矩形，再对该矩形做正向旋转
                rotated_building_shape = rotate(building_geometry, -rotate_angle - fit_angle, origin='centroid')
                building_envelope = rotated_building_shape.envelope
                building_geometry = rotate(building_envelope, rotate_angle, origin='centroid')
            else:
                # 距离边界过远的建筑，也不用旋转角度适应边界
                building_geometry = rotate(building_geometry, angle, origin='centroid')
        else:
            # 长宽比特别大的建筑，不需要适应基地边界方向，只要转动其本身的角度即可
            building_geometry = rotate(building_geometry, angle, origin='centroid')

        return building_geometry

    @staticmethod
    def get_line_list_cover_building_num(line_list, multi_buffered_building, min_cover_distance):
        if not line_list:
            return []

        # 如果建筑在路网环路内，也算覆盖 / 路网组成的多边形：
        road_ring_polygon = unary_union(list(polygonize(line_list)))
        road_ring_polygon_preped = prep(road_ring_polygon)

        # 可以被道路在一定距离内覆盖的建筑的 list
        road_under_distance_limit_building_idx_list = list()
        for building_idx, building in enumerate(multi_buffered_building.geoms):
            if road_ring_polygon_preped.contains(building):
                road_under_distance_limit_building_idx_list.append(building_idx)
                continue

            for line in line_list:
                dist = building.distance(line)
                if dist <= min_cover_distance and building_idx not in road_under_distance_limit_building_idx_list:
                    road_under_distance_limit_building_idx_list.append(building_idx)
                    break

        return road_under_distance_limit_building_idx_list

    def get_roads_cover_change(self, old_line_list, old_lines_to_drop, new_lines_to_add, multi_buffered_building,
                               threshold_percent=0.8):
        if threshold_percent != 1:
            threshold_percent = self.random_threshold_road_cover
        # 计算当路网发生改变时，新旧路网对建筑可达性的变化情况，如果该变化符合 threshold_percent,则当前的路网变化可以被接受
        min_distance_limit = self.min_distance_limit  # 判定道路是否覆盖该建筑的最小距离
        # 旧有路网覆盖的建筑集合
        old_road_cover_building_list = self.get_line_list_cover_building_num(
            line_list=old_line_list,
            multi_buffered_building=multi_buffered_building,
            min_cover_distance=min_distance_limit)

        # 删除该点以后，新的路网的组成：
        new_line_list = [line for line in old_line_list if line not in old_lines_to_drop] + new_lines_to_add
        new_road_cover_building_list = self.get_line_list_cover_building_num(
            line_list=new_line_list,
            multi_buffered_building=multi_buffered_building,
            min_cover_distance=min_distance_limit)

        # 建筑总数目
        old_cover_num = len(old_road_cover_building_list)
        new_cover_num = len(new_road_cover_building_list)
        total_num = len(multi_buffered_building.geoms)
        # 计算当路网发生改变时，新旧路网对建筑可达性的变化情况，如果该变化符合 threshold_percent,则当前的路网变化可以被接受
        return new_cover_num / total_num >= old_cover_num / total_num * threshold_percent

    @staticmethod
    def get_point_in_lines(lines):
        """
        point_in_lines = {point_coord: {'neighbour_point_list': [p1_coord, p2_coord],
                                        'neighbour_line_list':  [Line_1, Line_2]}}
        """
        point_in_lines = {}
        for line in lines:
            p1_coord, p2_coord = list(line.coords)[0:2]

            if p1_coord not in point_in_lines:
                point_in_lines[p1_coord] = {'neighbour_point_list': [], 'neighbour_line_list': []}
            if p2_coord not in point_in_lines:
                point_in_lines[p2_coord] = {'neighbour_point_list': [], 'neighbour_line_list': []}

            if p2_coord not in point_in_lines[p1_coord]['neighbour_point_list']:
                point_in_lines[p1_coord]['neighbour_point_list'].append(p2_coord)
                point_in_lines[p1_coord]['neighbour_line_list'].append(line)

            if p1_coord not in point_in_lines[p2_coord]['neighbour_point_list']:
                point_in_lines[p2_coord]['neighbour_point_list'].append(p1_coord)
                point_in_lines[p2_coord]['neighbour_line_list'].append(line)
        return point_in_lines

    def drop_point_far_from_building(self, lines, valid_area, multi_buffered_building, buffered_obstacle_geom):
        switch = True
        while switch:
            switch = False
            # 对当前路网建立点和道路的对应关系
            point_in_lines = self.get_point_in_lines(lines)
            for point_coord, point_val in point_in_lines.items():
                point = Point(point_coord)
                if len(point_val['neighbour_point_list']) == 2:
                    line_1, line_2 = point_val['neighbour_line_list']
                    # 计算该点到各个建筑的最小距离
                    distance = point.distance(multi_buffered_building)
                    if distance > 10:
                        new_line = LineString([point_in_lines[point_coord]['neighbour_point_list'][0],
                                               point_in_lines[point_coord]['neighbour_point_list'][1]])

                        # 如果新线段可以直接通过规范，则改变当前路网
                        roads_cover_change = self.get_roads_cover_change(
                            old_line_list=lines,
                            old_lines_to_drop=[line_1, line_2],
                            new_lines_to_add=[new_line],
                            multi_buffered_building=multi_buffered_building)

                        if roads_cover_change:
                            # 还要计算新增加道路是否在有效面积内 / 是否会碰撞到各个建筑
                            if (new_line.within(valid_area)
                                    and not new_line.intersects(multi_buffered_building)
                                    and not new_line.intersects(buffered_obstacle_geom)):
                                logger.info('drop point far from building')
                                # logger.info('drop:{} , {}; add:{}'.format(line_1, line_2, new_line))
                                # 如果同时符合道路可达性，并且新道路不会碰撞到建筑：
                                lines = self.get_new_road_lines(lines=lines,
                                                                old_lines_to_drop=[line_1, line_2],
                                                                new_lines_to_add=[new_line])
                                # 更改循环开关，因为当前改变可能会引起新的同类问题
                                switch = True
                                break
        return lines

    @staticmethod
    def get_new_road_lines(lines, old_lines_to_drop, new_lines_to_add):
        old_left_lines = [line for line in lines if line not in old_lines_to_drop]
        new_lines_to_add_without_duplicate = [line for line in new_lines_to_add if line not in old_left_lines]
        new_lines = old_left_lines + new_lines_to_add_without_duplicate
        return new_lines

    def drop_single_line(self, lines):
        # 删除路网中的死胡同 / 单一线段
        switch = True
        while switch:
            point_in_lines = self.get_point_in_lines(lines)
            switch = False
            for point_coord, point_val in point_in_lines.items():
                if len(point_val['neighbour_point_list']) == 1:
                    line_1 = point_val['neighbour_line_list'][0]
                    if line_1 in lines and len(lines) > 10:
                        lines.remove(line_1)
                        switch = True
                        break
        return lines

    def drop_acute_angle(self, lines, valid_area, multi_buffered_building, buffered_obstacle_geom):
        switch = True
        while switch:
            point_in_lines = self.get_point_in_lines(lines)
            switch = False
            for point_coord, point_val in point_in_lines.items():
                if len(point_val['neighbour_point_list']) == 2:
                    line_1 = point_val['neighbour_line_list'][0]
                    line_2 = point_val['neighbour_line_list'][1]

                    cos_v1_v2 = util.get_cos_with_two_line(line_1, line_2)

                    # 当角度为锐角时
                    if cos_v1_v2 > math.cos(math.pi / 2 * 80 / 90):

                        # 先考虑最简单的删除方式：直接连接锐角的两个边的顶点：
                        new_line = LineString([point_val['neighbour_point_list'][0],
                                               point_val['neighbour_point_list'][1]])

                        roads_cover_change = self.get_roads_cover_change(
                            old_line_list=lines,
                            old_lines_to_drop=[line_1, line_2],
                            new_lines_to_add=[new_line],
                            multi_buffered_building=multi_buffered_building)

                        if roads_cover_change:
                            if (new_line.within(valid_area)
                                    and not new_line.intersects(multi_buffered_building)
                                    and not new_line.intersects(buffered_obstacle_geom)):
                                logger.info('drop acute angle: connect two neighbour point')
                                lines = self.get_new_road_lines(lines=lines,
                                                                old_lines_to_drop=[line_1, line_2],
                                                                new_lines_to_add=[new_line])
                                switch = True
                                break

                        # 无效则考虑另一种连接方式：从短边的端点向长边做垂线
                        long_line, short_line = (line_1, line_2) if line_1.length > line_2.length else (line_2, line_1)
                        # 锐角两个边的端点
                        long_point = Point(next(coord for coord in long_line.coords if coord != point_coord))
                        short_point = Point(next(coord for coord in short_line.coords if coord != point_coord))
                        # 短边到长边最短距离的交点(其实就是垂足)：
                        tmp_nearest_points = nearest_points(short_point, long_line)
                        foot_point = [p for p in tmp_nearest_points if p != short_point][0]

                        new_vertical_line = LineString(tmp_nearest_points)
                        new_long_line = LineString([foot_point, long_point])
                        roads_cover_change = self.get_roads_cover_change(
                            old_line_list=lines,
                            old_lines_to_drop=[line_1, line_2],
                            new_lines_to_add=[new_vertical_line, new_long_line],
                            multi_buffered_building=multi_buffered_building)

                        if roads_cover_change:
                            if (new_vertical_line.within(valid_area)
                                    and new_long_line.within(valid_area)
                                    and not new_vertical_line.intersects(multi_buffered_building)
                                    and not new_vertical_line.intersects(buffered_obstacle_geom)
                                    and not new_long_line.intersects(multi_buffered_building)
                                    and not new_long_line.intersects(buffered_obstacle_geom)):
                                logger.info('drop acute angle: make vertical line against long line')
                                lines = self.get_new_road_lines(lines=lines,
                                                                old_lines_to_drop=[line_1, line_2],
                                                                new_lines_to_add=[new_long_line, new_vertical_line])
                                switch = True
                                break
        return lines

    def drop_right_angle(self, lines, valid_area, multi_buffered_building, buffered_obstacle_geom):
        # 去掉接近直角，但是边长有限的情况
        switch = True
        while switch:
            point_in_lines = self.get_point_in_lines(lines)
            switch = False
            for point_coord, point_val in point_in_lines.items():
                # 确保当前点只有两个邻边
                if len(point_val['neighbour_point_list']) == 2:
                    line_1 = point_val['neighbour_line_list'][0]
                    line_2 = point_val['neighbour_line_list'][1]

                    point_angle_cos = util.get_cos_with_two_line(line_1, line_2)

                    # 判定是否为直角 / 接近直角,同时其中一条边较短
                    if (math.cos(math.pi / 2 * 100 / 90) < point_angle_cos < math.cos(math.pi / 2 * 80 / 90)
                            and (line_1.length < 30 or line_2.length < 30)):

                        # 第一种修正方法，直接连接直角两个端点
                        new_line = LineString([point_val['neighbour_point_list'][0],
                                               point_val['neighbour_point_list'][1]])
                        # 计算删除该点对建筑可达性的影响
                        roads_cover_change = self.get_roads_cover_change(
                            old_line_list=lines,
                            old_lines_to_drop=[line_1, line_2],
                            new_lines_to_add=[new_line],
                            multi_buffered_building=multi_buffered_building)

                        if roads_cover_change:
                            if (new_line.within(valid_area)
                                    and not new_line.intersects(multi_buffered_building)
                                    and not new_line.intersects(buffered_obstacle_geom)):
                                logger.info('drop right angle: connect two neighbour point')
                                lines = self.get_new_road_lines(lines=lines,
                                                                old_lines_to_drop=[line_1, line_2],
                                                                new_lines_to_add=[new_line])
                                switch = True
                                break

                        # 第二种修正方法：将短边的临边进行延伸，和长边相交，截掉该直角
                        long_line, short_line = (line_1, line_2) if line_1.length > line_2.length else (line_2, line_1)

                        # 短边和长边的对应端点
                        long_point = Point(next(coord for coord in long_line.coords if coord != point_coord))
                        short_point = Point(next(coord for coord in short_line.coords if coord != point_coord))
                        if len(point_in_lines[(short_point.x, short_point.y)]['neighbour_line_list']) == 2 and len(
                                point_in_lines[(long_point.x, long_point.y)]['neighbour_line_list']) == 2:
                            # 计算该直角短边端点的另一个邻边
                            short_point_neighbour_line = [
                                line for line in point_in_lines[(short_point.x, short_point.y)]['neighbour_line_list']
                                if line != short_line][0]
                            # 计算该直角长边端点的另一个邻边
                            long_point_neighbour_line = [line for line in point_in_lines[(long_point.x, long_point.y)][
                                'neighbour_line_list'] if line != long_line][0]

                            # 沿短边临边做延长线，和长边所在直线相交
                            short_line_neighbour_cross_point_coord = util.get_cross_point_coord(
                                short_point_neighbour_line, long_line)
                            # 如果该交点存在：
                            if short_line_neighbour_cross_point_coord:
                                short_line_neighbour_cross_point = Point(short_line_neighbour_cross_point_coord)
                                # 如果该交点在长边上(而不是在长边的延长线上)
                                if (short_line_neighbour_cross_point.distance(long_line) < 0.1
                                        and short_line_neighbour_cross_point.distance(Point(point_coord)) > 1):

                                    new_point = short_line_neighbour_cross_point
                                    new_cross_line = LineString([new_point, short_point])
                                    new_long_line = LineString([new_point, long_point])

                                    roads_cover_change = self.get_roads_cover_change(
                                        old_line_list=lines,
                                        old_lines_to_drop=[line_1, line_2],
                                        new_lines_to_add=[new_cross_line, new_long_line],
                                        multi_buffered_building=multi_buffered_building)

                                    if roads_cover_change:
                                        if (new_cross_line.within(valid_area)
                                                and new_long_line.within(valid_area)
                                                and not new_cross_line.intersects(multi_buffered_building)
                                                and not new_cross_line.intersects(buffered_obstacle_geom)
                                                and not new_long_line.intersects(multi_buffered_building)
                                                and not new_long_line.intersects(buffered_obstacle_geom)):
                                            logger.info('drop right angle:extend short line neighbour across long line')
                                            lines = self.get_new_road_lines(
                                                lines=lines,
                                                old_lines_to_drop=[line_1, line_2],
                                                new_lines_to_add=[new_cross_line, new_long_line])
                                            switch = True
                                            break

                            # 如果以上方法依然无法消除该直角，尝试从长边临边向短边做延长线
                            long_line_neighbour_cross_point_coord = util.get_cross_point_coord(
                                long_point_neighbour_line, short_line)
                            if long_line_neighbour_cross_point_coord:
                                long_line_neighbour_cross_point = Point(long_line_neighbour_cross_point_coord)
                                if (long_line_neighbour_cross_point.distance(short_line) < 0.1
                                        and long_line_neighbour_cross_point.distance(Point(point_coord)) > 1):
                                    new_point = long_line_neighbour_cross_point
                                    new_cross_line = LineString([new_point, long_point])
                                    new_short_line = LineString([new_point, short_point])

                                    roads_cover_change = self.get_roads_cover_change(
                                        old_line_list=lines,
                                        old_lines_to_drop=[line_1, line_2],
                                        new_lines_to_add=[new_cross_line, new_short_line],
                                        multi_buffered_building=multi_buffered_building)
                                    if roads_cover_change:
                                        if (new_cross_line.within(valid_area)
                                                and new_short_line.within(valid_area)
                                                and not new_cross_line.intersects(multi_buffered_building)
                                                and not new_cross_line.intersects(buffered_obstacle_geom)
                                                and not new_short_line.intersects(multi_buffered_building)
                                                and not new_short_line.intersects(buffered_obstacle_geom)):
                                            logger.info('drop right angle:extend long line neighbour across short line')
                                            lines = self.get_new_road_lines(
                                                lines=lines,
                                                old_lines_to_drop=[line_1, line_2],
                                                new_lines_to_add=[new_cross_line, new_short_line])
                                            switch = True
                                            break
        return lines

    def drop_s_corner(self, lines, valid_area, multi_buffered_building, buffered_obstacle_geom):
        # 删除路网中的 S 弯 和 U 型弯

        switch = True
        while switch:
            switch = False
            min_us_curve_limit = 50  # 如果该道路长度小于50，则检测是否满足 U / S 弯 [之所以长度较大，为了除去部分凸出的U型道路]
            point_in_lines = self.get_point_in_lines(lines)

            for line in lines:
                p1_coord = tuple(line.coords[0])
                p2_coord = tuple(line.coords[1])

                # 如果该线段的两个端点都拥有2个临边(避免出现断点和岔口情况)
                if (line.length < min_us_curve_limit
                        and len(point_in_lines[p1_coord]['neighbour_line_list']) == 2
                        and len(point_in_lines[p2_coord]['neighbour_line_list']) == 2):
                    # 两个邻边
                    p1_line = next(neighbor_line for neighbor_line
                                   in point_in_lines[p1_coord]['neighbour_line_list'] if neighbor_line != line)
                    p2_line = next(neighbor_line for neighbor_line
                                   in point_in_lines[p2_coord]['neighbour_line_list'] if neighbor_line != line)
                    # 中间线段端点之外的两个邻接点
                    p1_neighbour_point_coord = next(p for p in point_in_lines[p1_coord]['neighbour_point_list']
                                                    if p != p2_coord)
                    p2_neighbour_point_coord = next(p for p in point_in_lines[p2_coord]['neighbour_point_list']
                                                    if p != p1_coord)

                    # 计算两个邻接点所在角的 cos 值
                    p1_cos = util.get_cos_with_two_line(line, p1_line)
                    p2_cos = util.get_cos_with_two_line(line, p2_line)

                    # 如果2个转角都接近于 90 度， 判定当前连续转角是 U 型还是 S 型：
                    if (math.cos(math.pi / 2 * 120 / 90) < p1_cos < math.cos(math.pi / 2 * 80 / 90)
                            and math.cos(math.pi / 2 * 120 / 90) < p2_cos < math.cos(math.pi / 2 * 80 / 90)):
                        # 构造中间线段邻接点连线，用来判定是 S / U 型弯
                        # 如果该连线和中间的线段有交点，则为 S 弯， 否则为 U 型弯
                        us_judge_assist_line = LineString([p1_neighbour_point_coord, p2_neighbour_point_coord])
                        if us_judge_assist_line.intersects(line):
                            # 尝试从 p2 向 p1 neighbour point 连线
                            new_line = LineString([p2_coord, p1_neighbour_point_coord])
                            roads_cover_change = self.get_roads_cover_change(
                                old_line_list=lines,
                                old_lines_to_drop=[line, p1_line],
                                new_lines_to_add=[new_line],
                                multi_buffered_building=multi_buffered_building)

                            if roads_cover_change:
                                if (new_line.within(valid_area)
                                        and not new_line.intersects(multi_buffered_building)
                                        and not new_line.intersects(buffered_obstacle_geom)):
                                    logger.info('drop S corner:connect p1 with p2 neighbour point')
                                    # logger.info('drop:{} , {}; add:{}'.format(line, p1_line, new_line))
                                    lines = self.get_new_road_lines(lines=lines,
                                                                    old_lines_to_drop=[line, p1_line],
                                                                    new_lines_to_add=[new_line])
                                    switch = True
                                    break

                            # 尝试从 p1 向 p2 neighbour 连线
                            new_line = LineString([p1_coord, p2_neighbour_point_coord])
                            roads_cover_change = self.get_roads_cover_change(
                                old_line_list=lines,
                                old_lines_to_drop=[line, p2_line],
                                new_lines_to_add=[new_line],
                                multi_buffered_building=multi_buffered_building)

                            if roads_cover_change:
                                if (new_line.within(valid_area)
                                        and not new_line.intersects(multi_buffered_building)
                                        and not new_line.intersects(buffered_obstacle_geom)):
                                    logger.info('drop S corner:connect p2 with p1 neighbour point')
                                    # logger.info('drop:{} , {}; add:{}'.format(line, p2_line, new_line))
                                    lines = self.get_new_road_lines(lines=lines,
                                                                    old_lines_to_drop=[line, p2_line],
                                                                    new_lines_to_add=[new_line])
                                    switch = True
                                    break

                            # 尝试从 p1 向 p2_line 上的等分点做连线：
                            tmp_lines_res = self.connect_point_to_its_neighbour_line_divided_point(
                                lines=lines,
                                valid_area=valid_area,
                                multi_buffered_building=multi_buffered_building,
                                buffered_obstacle_geom=buffered_obstacle_geom,
                                long_point=p2_neighbour_point_coord,
                                short_point=p1_coord,
                                middle_point=p2_coord,
                                line_1=line,
                                line_2=p2_line)
                            if tmp_lines_res:
                                lines = tmp_lines_res
                                switch = True
                                break

                            # 尝试从 p2 向 p1_line 上的等分点做连线
                            tmp_lines_res = self.connect_point_to_its_neighbour_line_divided_point(
                                lines=lines,
                                valid_area=valid_area,
                                multi_buffered_building=multi_buffered_building,
                                buffered_obstacle_geom=buffered_obstacle_geom,
                                long_point=p1_neighbour_point_coord,
                                short_point=p2_coord,
                                middle_point=p1_coord,
                                line_1=line,
                                line_2=p1_line)
                            if tmp_lines_res:
                                lines = tmp_lines_res
                                switch = True
                                break

                        else:  # 如果是 U 型弯的情况下：
                            # 尝试直接删除整个 U 型弯：
                            new_line = LineString([p1_neighbour_point_coord, p2_neighbour_point_coord])
                            roads_cover_change = self.get_roads_cover_change(
                                old_line_list=lines,
                                old_lines_to_drop=[line, p1_line, p2_line],
                                new_lines_to_add=[new_line],
                                multi_buffered_building=multi_buffered_building)

                            if roads_cover_change:
                                if (new_line.within(valid_area)
                                        and not new_line.intersects(multi_buffered_building)
                                        and not new_line.intersects(buffered_obstacle_geom)):
                                    logger.info('drop U corner: connect p1 neighbour point with p1 neighbour point')
                                    lines = self.get_new_road_lines(lines=lines,
                                                                    old_lines_to_drop=[line, p1_line, p2_line],
                                                                    new_lines_to_add=[new_line])
                                    switch = True
                                    break

                            # 从短边端点到长边投影(垂线)，连接短边端点和投影点(垂足)
                            if p1_line.length > p2_line.length:
                                tmp_nearest_points = nearest_points(Point(p2_neighbour_point_coord), p1_line)
                                foot_point = next(p for p in tmp_nearest_points if p != Point(p2_neighbour_point_coord))
                                new_middle_line = LineString(tmp_nearest_points)
                                new_side_line = LineString([foot_point, Point(p1_neighbour_point_coord)])
                            else:
                                tmp_nearest_points = nearest_points(Point(p1_neighbour_point_coord), p2_line)
                                foot_point = next(p for p in tmp_nearest_points if p != Point(p1_neighbour_point_coord))
                                new_middle_line = LineString(tmp_nearest_points)
                                new_side_line = LineString([foot_point, Point(p2_neighbour_point_coord)])

                            roads_cover_change = self.get_roads_cover_change(
                                old_line_list=lines,
                                old_lines_to_drop=[line, p1_line, p2_line],
                                new_lines_to_add=[new_middle_line, new_side_line],
                                multi_buffered_building=multi_buffered_building)
                            if roads_cover_change:
                                if (new_middle_line.within(valid_area)
                                        and new_side_line.within(valid_area)
                                        and not new_middle_line.intersects(multi_buffered_building)
                                        and not new_middle_line.intersects(buffered_obstacle_geom)
                                        and not new_side_line.intersects(multi_buffered_building)
                                        and not new_side_line.intersects(buffered_obstacle_geom)):
                                    logger.info('drop U corner: make vertical line across long line from short point')
                                    lines = self.get_new_road_lines(lines=lines,
                                                                    old_lines_to_drop=[line, p1_line, p2_line],
                                                                    new_lines_to_add=[new_middle_line, new_side_line])
                                    switch = True
                                    break
        return lines

    def drop_short_line(self, lines, valid_area, multi_buffered_building, buffered_obstacle_geom):
        switch = True
        while switch:
            point_in_lines = self.get_point_in_lines(lines)
            switch = False
            for point_coord, point_val in point_in_lines.items():
                if len(point_val['neighbour_point_list']) != 2:
                    continue

                line_1, line_2 = point_val['neighbour_line_list'][0:2]
                if line_1.length > 12 and line_2.length > 12:
                    continue

                # 尝试直接连接该点的两个邻接端点
                new_line = LineString([point_val['neighbour_point_list'][0],
                                       point_val['neighbour_point_list'][1]])
                roads_cover_change = self.get_roads_cover_change(old_line_list=lines,
                                                                 old_lines_to_drop=[line_1, line_2],
                                                                 new_lines_to_add=[new_line],
                                                                 multi_buffered_building=multi_buffered_building)
                if roads_cover_change:
                    if (new_line.within(valid_area)
                            and not new_line.intersects(multi_buffered_building)
                            and not new_line.intersects(buffered_obstacle_geom)):
                        logger.info('drop short line：connect neighbour point with short line far point')
                        # logger.info('drop:{} , {} ; add:{} '.format(line_1, line_2, new_line))
                        lines = self.get_new_road_lines(lines=lines,
                                                        old_lines_to_drop=[line_1, line_2],
                                                        new_lines_to_add=[new_line])
                        switch = True
                        break

                # 如果无法直接将两条边的端点相连，则尝试将短边截掉
                long_line, short_line = (line_1, line_2) if line_1.length > line_2.length else (line_2, line_1)
                long_point_coord = next(coord for coord in long_line.coords if coord != point_coord)
                short_point_coord = next(coord for coord in short_line.coords if coord != point_coord)

                if (len(point_in_lines[short_point_coord]['neighbour_line_list']) == 2 and
                        len(point_in_lines[long_point_coord]['neighbour_line_list']) == 2):
                    short_point_neighbour_line = \
                        next(neighbor_line for neighbor_line
                             in point_in_lines[short_point_coord]['neighbour_line_list'] if neighbor_line != short_line)

                    # 试图从短边的临边做延长线：
                    short_line_neighbour_cross_point_coord = util.get_cross_point_coord(short_point_neighbour_line,
                                                                                        long_line)
                    if short_line_neighbour_cross_point_coord:

                        if Point(short_line_neighbour_cross_point_coord).distance(long_line) < 0.1 and Point(
                                short_line_neighbour_cross_point_coord).distance(Point(point_coord)) > 1:

                            new_point = short_line_neighbour_cross_point_coord
                            new_cross_line = LineString([new_point, short_point_coord])
                            new_long_line = LineString([new_point, long_point_coord])

                            roads_cover_change = self.get_roads_cover_change(
                                old_line_list=lines,
                                old_lines_to_drop=[line_1, line_2],
                                new_lines_to_add=[new_cross_line,
                                                  new_long_line],
                                multi_buffered_building=multi_buffered_building)
                            if roads_cover_change:
                                if (new_cross_line.within(valid_area)
                                        and new_long_line.within(valid_area)
                                        and not new_cross_line.intersects(multi_buffered_building)
                                        and not new_cross_line.intersects(buffered_obstacle_geom)
                                        and not new_long_line.intersects(multi_buffered_building)
                                        and not new_long_line.intersects(buffered_obstacle_geom)):
                                    logger.info('drop short line：'
                                                'extend its neighbour line across another neighbour line')
                                    lines = self.get_new_road_lines(lines=lines,
                                                                    old_lines_to_drop=[line_1, line_2],
                                                                    new_lines_to_add=[new_cross_line, new_long_line])
                                    switch = True
                                    break

                    # 如果延长线无法起作用，则尝试从短边端点向长边的不同等分点连线
                    tmp_lines_res = self.connect_point_to_its_neighbour_line_divided_point(
                        lines=lines,
                        valid_area=valid_area,
                        multi_buffered_building=multi_buffered_building,
                        buffered_obstacle_geom=buffered_obstacle_geom,
                        long_point=long_point_coord,
                        short_point=short_point_coord,
                        middle_point=point_coord,
                        line_1=line_1,
                        line_2=line_2)
                    if tmp_lines_res:
                        lines = tmp_lines_res
                        switch = True
                        break

        return lines

    def connect_point_to_its_neighbour_line_divided_point(self, lines, valid_area, multi_buffered_building,
                                                          buffered_obstacle_geom, long_point, short_point,
                                                          middle_point, line_1, line_2):
        for idx in range(2, 10):
            delta_x = long_point[0] - middle_point[0]
            delta_y = long_point[1] - middle_point[1]
            new_point_coord = [middle_point[0] + delta_x / idx, middle_point[1] + delta_y / idx]

            new_line_1 = LineString([short_point, new_point_coord])
            new_line_2 = LineString([long_point, new_point_coord])
            roads_cover_change = self.get_roads_cover_change(old_line_list=lines,
                                                             old_lines_to_drop=[line_1, line_2],
                                                             new_lines_to_add=[new_line_1, new_line_2],
                                                             multi_buffered_building=multi_buffered_building)
            if roads_cover_change:
                if (new_line_1.within(valid_area)
                        and new_line_2.within(valid_area)
                        and not new_line_1.intersects(multi_buffered_building)
                        and not new_line_2.intersects(multi_buffered_building)
                        and not new_line_1.intersects(buffered_obstacle_geom)
                        and not new_line_2.intersects(buffered_obstacle_geom)):
                    logger.info('drop short line：connect one point across long neighbour line divided point')
                    lines = self.get_new_road_lines(lines=lines, old_lines_to_drop=[line_1, line_2],
                                                    new_lines_to_add=[new_line_1, new_line_2])
                    return lines
        return None

    def drop_useless_point(self, lines, valid_area, multi_buffered_building, buffered_obstacle_geom):
        # 为了保证尽量直线为主，删除部分对建筑可达性没有影响，但是会造成路网不直的拐点
        switch = True
        while switch:
            switch = False
            point_in_lines = self.get_point_in_lines(lines)
            for point_coord, point_val in point_in_lines.items():
                if len(point_val['neighbour_point_list']) == 2:
                    line_1 = point_val['neighbour_line_list'][0]
                    line_2 = point_val['neighbour_line_list'][1]

                    new_line = LineString([point_in_lines[point_coord]['neighbour_point_list'][0],
                                           point_in_lines[point_coord]['neighbour_point_list'][1]])
                    # 要求对路网可达性完全没有负面影响，所以这里的 threshold percent = 1
                    roads_cover_change = self.get_roads_cover_change(old_line_list=lines,
                                                                     old_lines_to_drop=[line_1, line_2],
                                                                     new_lines_to_add=[new_line],
                                                                     multi_buffered_building=multi_buffered_building,
                                                                     threshold_percent=1)

                    if roads_cover_change:
                        if (new_line.within(valid_area)
                                and not new_line.intersects(multi_buffered_building)
                                and not new_line.intersects(buffered_obstacle_geom)):
                            logger.info('drop useless point：connect its two neighbour point')
                            lines = self.get_new_road_lines(lines=lines,
                                                            old_lines_to_drop=[line_1, line_2],
                                                            new_lines_to_add=[new_line])
                            switch = True
                            break
        return lines

    def drop_error_line(self, lines, valid_area, multi_buffered_building, obstacle_geom):
        lines_list = []
        prev_line_len = len(lines)
        for i in range(20):
            # 去掉距离建筑远的无用的点
            lines = self.drop_point_far_from_building(lines, valid_area, multi_buffered_building, obstacle_geom)
            lines_list.append(lines)
            # 去掉单一线段(死胡同情况)
            lines = self.drop_single_line(lines)
            lines_list.append(lines)
            # # 去掉特别明显的锐角
            lines = self.drop_acute_angle(lines, valid_area, multi_buffered_building, obstacle_geom)
            lines_list.append(lines)
            # 去掉接近直角，但是边长有限的情况
            # lines = self.drop_right_angle(lines, valid_area, multi_buffered_building)
            # lines_list.append(lines)
            # # 合并/去掉 短路
            lines = self.drop_short_line(lines, valid_area, multi_buffered_building, obstacle_geom)
            lines_list.append(lines)
            # 去掉短边两个端点急转弯的情况
            lines = self.drop_s_corner(lines, valid_area, multi_buffered_building, obstacle_geom)
            lines_list.append(lines)
            if len(lines) == prev_line_len:
                break
            prev_line_len = len(lines)

        # 去掉无用的点，将道路尽可能的拉直
        for i in range(2):
            lines = self.drop_useless_point(lines, valid_area, multi_buffered_building, obstacle_geom)
        lines_list.append(lines)
        # 拉直过程中会出现部分锐角和小直角，还有一些不合规范的短道路，再进行二次清理
        for i in range(5):
            lines = self.drop_acute_angle(lines, valid_area, multi_buffered_building, obstacle_geom)
            lines = self.drop_right_angle(lines, valid_area, multi_buffered_building, obstacle_geom)
            lines = self.drop_short_line(lines, valid_area, multi_buffered_building, obstacle_geom)
        lines_list.append(lines)
        return lines_list

    def generate_roads(self):
        layout = self.plan_to_layout()
        building_box_geom_list, obstacle_box_geom_list, building_geom_list, obstacle_geom_list = layout
        valid_area = self.find_valid_area(base_geom=self.base,
                                          building_geom_list=building_box_geom_list,
                                          obstacle_geom_list=obstacle_box_geom_list)

        point_list = self.building_geom_list_to_point_geom_list(
            building_geom_list=building_box_geom_list,
            valid_area=valid_area,
            obstacle_geom_list=obstacle_geom_list)

        final_lines_list = self.find_and_connect_marginal_points(
            points=point_list,
            valid_area=valid_area,
            building_geom_list=building_geom_list,
            obstacle_geom_list=obstacle_geom_list)
        lines = final_lines_list[-1]
        logger.info('成功生成道路主干')
        # # 增加验证：
        # road_ring_polygon = unary_union(list(polygonize(lines)))
        # # 所有的建筑轮廓
        # building_multipolygon = MultiPolygon(obstacle_geom_list + building_geom_list).buffer(
        #     self.road_width / 2, cap_style=1, join_style=2)
        # print(list(polygonize(lines)))
        # road_linestring_polygon = Point(road_ring_polygon.centroid)

        #road_linestring_polygon = Polygon(list(polygonize(lines)))
        # logger.info('正在进行平滑处理')
        # while (road_linestring_polygon.area <= road_ring_polygon.area * 0.8
        #        or road_linestring_polygon.boundary.intersects(building_multipolygon)
        #        or type(road_linestring_polygon) == MultiPolygon) and self.road_buffer_distance >= 5:
        #     # 通过先向内 buffer 后向外 buffer 可以产生凸出部分的平滑
        #     road_polygon_outer_buffer = road_ring_polygon.buffer(
        #         -self.road_buffer_distance, resolution=6, cap_style=1, join_style=1, mitre_limit=10.0)
        #     road_polygon_inner_buffer = road_polygon_outer_buffer.buffer(
        #         self.road_buffer_distance + self.road_buffer_distance,
        #         resolution=6, cap_style=1, join_style=1, mitre_limit=10.0)
        #     # 通过向内求 parallel offset 得到凹陷部分的平滑效果
        #     print(type(road_polygon_inner_buffer))
        #     road_linestring = road_polygon_inner_buffer.boundary.parallel_offset(-self.road_buffer_distance,
        #                                                                          side=1, resolution=6, join_style=1)
        #     road_linestring_polygon = Polygon(road_linestring.coords)
        #
        #     self.road_buffer_distance -= 1
        # road_polygon_area_ratio = round((road_linestring_polygon.area / self.base.area), 3)
        # 只有当生成的环线围成的面积占基地面积的百分比大于 0.08 时, 该道路才会放出来：
        # if road_polygon_area_ratio >= 0.1:
        #     logger.info('成功生成道路')
        #     road_geom = road_linestring_polygon.boundary.buffer(self.road_width / 2)
        # else:
        #     print(road_polygon_area_ratio)
        #     print('道路占比不达标')
        #     logger.info('道路占比不达标')
        #     road_geom = None
        # === 道路环线生成过程演示,用于调试 ===
        # import matplotlib.pyplot as plt
        # import geopandas as gpd
        # import scipy as sp
        # import matplotlib.colors as colors
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.set_yticks([])
        # ax.set_xticks([])
        #
        # #gpd.GeoSeries(point_list).plot(ax=ax, color='green', alpha=0.5)
        # gpd.GeoSeries([self.base]).plot(ax=ax, color='blue')
        # print(self.base.area)
        # #gpd.GeoSeries([road_geom]).plot(ax=ax, color='black', alpha=0.3)
        # gpd.GeoSeries([building_multipolygon]).plot(ax=ax, color='red')
        # #gpd.GeoSeries([valid_area]).plot(ax=ax, color='red', alpha=0.4)
        #
        # buffered_boxs = [building.buffer(self.building_buffer_distance, join_style=2)
        #                  for building in building_box_geom_list]
        #
        # #gpd.GeoSeries(buffered_boxs).plot(ax=ax, color='red', alpha=0.5)
        # gpd.GeoSeries(building_box_geom_list).plot(ax=ax, color='yellow', alpha=0.5)
        #
        # #gpd.GeoSeries([building_multipolygon]).plot(ax=ax, color='blue', alpha=0.2)
        # # for lines in final_lines_list:
        # gpd.GeoSeries(final_lines_list[-1]).plot(ax=ax, color='green', linewidth=2.0)
        # plt.show()
        # ===

        return lines

    def is_fit_to_generate_circle_roads(self):
        """
        判断该基地是否适合使用该算法生成环形道路
        :return:
        """
        layout = self.plan_to_layout()
        building_box_geom_list, obstacle_box_geom_list, building_geom_list, obstacle_geom_list = layout

        # 计算该基地最小 bounding box
        base_min_box = self.base.minimum_rotated_rectangle

        base_border_line_list = []
        base_point_coords = list(base_min_box.exterior.coords)
        for idx in range(len(base_point_coords) - 1):
            line = LineString([base_point_coords[idx], base_point_coords[idx + 1]])
            base_border_line_list.append(line)

        # 用于计算基地的box长宽比
        short_line = min(base_border_line_list, key=lambda x: x.length)
        long_line = max(base_border_line_list, key=lambda x: x.length)

        # 计算建筑面积占比（建筑密度）：
        building_area = sum([building.area for building in building_geom_list] +
                            [obs.area for obs in obstacle_geom_list])

        res = True
        # 如果建筑数目小于5
        if len(building_geom_list) < 5:
            res = False
        # 如果基地面积占最小 bounding box 面积的比例低于 80%
        # 异形地块，但是形状适中，建筑密度高于一定值无法排布
        elif 0.65 < self.base.area / base_min_box.area < 0.7 and building_area / self.base.area > 0.12:
            res = False
        # 异形地块，形状非常离谱，无法排布
        elif self.base.area / base_min_box.area < 0.65:
            res = False
        # 接近长条形基地，不适合形成环路
        elif long_line.length / short_line.length > 1.5 and self.base.area < 35000:
            res = False
        # 接近长条形基地，基地面积有限且建筑比率过高
        elif (long_line.length / short_line.length > 2
              and self.base.area < 80000 and building_area / self.base.area > 0.1):
            res = False
        # 接近长条形基地且建筑比率过高
        elif long_line.length / short_line.length > 2 and building_area / self.base.area > 0.12:
            res = False
        elif long_line.length / short_line.length > 3:
            res = False

        return res
