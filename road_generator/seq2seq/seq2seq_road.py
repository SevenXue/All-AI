import numpy as np
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from seq2seq.models import AttentionSeq2Seq
from shapely.geometry import Polygon, LineString, Point
from matplotlib import pyplot as plt
from geopandas import GeoSeries
import cv2
import os

os.makedirs('model', exist_ok=True)
os.makedirs('tensorboard', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

TRAIN_URL = 'datasets/train_data.txt'
TEST_URL = 'datasets/test_data.txt'
model_file = 'model/s2s.h5'

def reset_endpoints(points):
    '''
        道路去零，并首尾相连
    :param points: list
    :return: points
    '''
    endpoints = []
    for i in range(len(points)):
        if points[i][0] != 0:
            endpoints.append(points[i])
    if len(endpoints) != 0:
        endpoints.append(points[0])
    return endpoints

def distance(x, y):
    '''
        两点距离
    :param x: list
    :param y: list
    :return: float
    '''
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2) ** 0.5

def rotate(point, cos, sin):
    '''
        坐标旋转
    :param point: list,原始坐标
    :param cos: int
    :param sin: int
    :return: list
    '''
    rotate_x = point[0] * cos + point[1] * sin
    rotate_y = point[1] * cos - point[0] * sin
    return [rotate_x, rotate_y]

def re_rotate(point, cos, sin, center):
    '''

    :param point:
    :param cos:
    :param sin:
    :return:
    '''
    re_x = point[0] * cos - point[1] * sin + center[0]
    re_y = point[1] * cos + point[0] * sin + center[1]
    return [re_x, re_y]

class Seq2seq():

    def __init__(self):
        # self.epochs = epochs
        self.latent_dim = 256
        self.max_encoder_length = 49
        self.max_decoder_length = 17

        # seq2seq
        self.model = AttentionSeq2Seq(batch_input_shape=(None, self.max_encoder_length, 2), hidden_dim=self.latent_dim, output_dim=2,
                                      output_length=self.max_decoder_length, depth=3)

        self.model.compile(loss='mse', loss_weights=[200], optimizer='rmsprop')

        if os.path.exists(model_file):
            self.model.load_weights(model_file)
        else:
            print('Model needs to be trained.')

    @staticmethod
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

    def set_x(self, url):
        """
            对input进行预处理
        :param url: str
        :return: np.array
        """
        inputs_data = []

        slope = []
        inter = []
        with open(url, 'r') as plan:
            pl = plan.readlines()

            self.multi_bases = []
            self.multi_buildings = []
            for line in pl:
                data = eval(line.strip('\n'))
                self.id = data['id']

                # 用于max_min归一化
                base_shapes = data['block']
                self.multi_bases.append(base_shapes)
                x_coords, y_coords = zip(*base_shapes)
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                slope.append([x_max - x_min, y_max - y_min])
                inter.append([x_min, y_min])

                # 对建筑进行中心化处理
                buildings = data['buildings']
                self.multi_buildings.append(buildings)
                center_buildings = []
                for building in buildings:
                    x, y = zip(*building)
                    center_buildings.append([np.mean(x), np.mean(y)])

                # 逆时针排序
                center_buildings = self.sort_paths(center_buildings)

                # 对建筑进行归一化处理
                for center in center_buildings:
                    center[0] = (center[0] - x_min) / (x_max - x_min)
                    center[1] = (center[1] - y_min) / (y_max - x_min)
                inputs_data.append(center_buildings)

                # # 对建筑进行外接 + 归一化处理
                # self.buildings = data['buildings']
                # center_buildings = []
                # for building in self.buildings:
                #     rect = cv2.minAreaRect(np.array(building, np.int32))
                #     box = cv2.boxPoints(rect)
                #     for i in range(4):
                #         x = (box[i][0] - x_min) / (x_max - x_min)
                #         y = (box[i][1] - y_min) / (y_max - y_min)
                #         center_buildings.append([x, y])

        num = len(inputs_data)
        self.max_encoder_length = max([max([len(inputs) for inputs in inputs_data]), self.max_encoder_length])

        print('Number of data:', num)
        print('Max length for inputs:', self.max_encoder_length)

        # 数据格式化处理
        encoder_data = np.zeros((num, self.max_encoder_length, 2))
        for i in range(num):
            for j in range(len(inputs_data[i])):
                encoder_data[i, j] = inputs_data[i][j]

        return encoder_data, np.array(slope), np.array(inter)

    def set_y(self, url):
        '''
            对output进行预处理
        :param url: str,
        :return: np.array
        '''
        output_data = []
        with open(url, 'r') as plan:
            pl = plan.readlines()

            for line in pl:
                data = eval(line.strip('\n'))

                # 用于max_min归一化
                base_shapes = data['block']
                x_coords, y_coords = zip(*base_shapes)
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)

                # 对道路进行逆时针排序 + 归一化处理
                endpoints = []
                self.lines = data['lines']
                # 逆时针排序
                self.lines = self.sort_paths(self.lines)

                # 归一化处理
                for path in self.lines:
                    x = (path[0] - x_min) / (x_max - x_min)
                    y = (path[1] - y_min) / (y_max - y_min)
                    endpoints.append([x, y])
                output_data.append(endpoints)

            self.max_decoder_length = max([max([len(outputs) for outputs in output_data]), self.max_decoder_length])
            print('Max length for outputs:', self.max_decoder_length)

            # 对道路数据进行格式化处理
            num = len(output_data)
            decoder_data = np.zeros((num, self.max_decoder_length, 2))
            for i in range(num):
                for k in range(len(output_data[i])):
                    decoder_data[i, k] = output_data[i][k]
        return decoder_data

    def reset_data(self, datasets, slope, inter):
        '''
            数据还原
        :param datasets:
        :param slope: w
        :param inter: b
        :return: datasets
        '''

        for i in range(len(datasets)):
            for j in range(len(datasets[i])):
                # 去除异常点
                if datasets[i][j][0] <= 0.05 or datasets[i][j][1] <= 0.05:
                    datasets[i][j][:] = 0
                else:
                    # 缩放
                    datasets[i][j] = datasets[i][j] * slope[i] + inter[i]
        return datasets

    def train(self, data_url, epochs):
        if os.path.exists(data_url):
            encoder_data, slope, inter = self.set_x(data_url)
            decoder_data = self.set_y(data_url)
            self.model.fit(encoder_data, decoder_data, batch_size=8, epochs=epochs,
                           callbacks=[TensorBoard(log_dir='tensorboard/seq_6')])
            self.model.save_weights(model_file)
        else:
            raise ValueError('data is not existing.')

    def predict(self, pre_data):
        '''
            预测
        :param pre_data:array,
        :return:
        '''
        self.model.load_weights(model_file)
        result = self.model.predict(pre_data)
        return result

    def data_post_process(self, points, buildings):
        '''
            道路数据后处理
        :param points: list,
        :param buildings: list
        :return: list, points
        '''

        label = 0
        list_of_adjusted_point = []
        for i in range(len(points) - 1):
            for building in buildings:
                rect = cv2.minAreaRect(np.array(building, np.int32))
                box = cv2.boxPoints(rect)
                center_x = sum([box[k][0] for k in range(4)]) / 4
                center_y = sum([box[k][1] for k in range(4)]) / 4
                cos = (box[0][0] - box[1][0]) / distance(box[0], box[1])
                sin = (box[0][1] - box[1][1]) / distance(box[0], box[1])

                rotate_box = rotate([box[0][0] - center_x, box[0][1] - center_y], cos, sin)
                radius_x = abs(rotate_box[0])
                radius_y = abs(rotate_box[1])

                def adjust_inter_point(point, size):
                    '''
                        处理点在建筑内的情况
                    :param point: tuple
                    :param size: int
                    :return: tuple, point
                    '''
                    point = rotate([point[0] - center_x, point[1] - center_y], cos, sin)
                    x = point[0]
                    y = point[1]
                    if radius_x - abs(x) < radius_y - abs(y):
                        if x < 0:
                            point = re_rotate([- radius_x - size, y], cos, sin, [center_x, center_y])
                        else:
                            point = re_rotate([radius_x + size, y], cos, sin, [center_x, center_y])
                    else:
                        if y < 0:
                            point = re_rotate([x, -radius_y - size], cos, sin, [center_x, center_y])
                        else:
                            point = re_rotate([x, radius_y + size], cos, sin, [center_x, center_y])
                    return point

                def adjust_outer_point(i, start_point, end_point, size):
                    '''
                        处理道路覆盖建筑的情况
                    :param i: int
                    :param start_point: tuple, 起始点
                    :param end_point: tuple，终止点
                    :param size: int
                    :return: None
                    '''
                    tmp = 0
                    start_point = rotate([start_point[0] - center_x, start_point[1] - center_y], cos, sin)
                    start_x = start_point[0]
                    start_y = start_point[1]
                    end_point = rotate([end_point[0] - center_x, end_point[1] - center_y], cos, sin)
                    end_x = end_point[0]
                    end_y = end_point[1]

                    # 增加新的点
                    if abs(start_x) > radius_x and abs(end_x) > radius_x and abs(start_y) > radius_y and abs(end_y) > radius_y:
                        print('add time!')
                        tmp += 1
                        k = (end_y - start_y) / (end_x - start_x)
                        path = LineString([(start_x, start_y), (end_x, end_y)])
                        if k > 0:
                            if Point([-radius_x, radius_y]).distance(path) < Point([radius_x, -radius_y]).distance(path):
                                points.insert(i+1, re_rotate([-radius_x - size, radius_y + size], cos, sin, [center_x, center_y]))
                            else:
                                points.insert(i+1, re_rotate([radius_x + size, -radius_y - size], cos, sin, [center_x, center_y]))
                        else:
                            if Point([radius_x, radius_y]).distance(path) < Point([-radius_x, -radius_y]).distance(path):
                                points.insert(i+1, re_rotate([radius_x + size, radius_y + size], cos, sin, [center_x, center_y]))
                            else:
                                points.insert(i+1, re_rotate([-radius_x - size, -radius_y - size], cos, sin, [center_x, center_y]))
                        return points[i], points[i+1], tmp

                    # 调整点
                    else:
                        ad_start_x = radius_x - abs(start_x) if radius_x - abs(start_x) > 0 else 10000
                        ad_start_y = radius_y - abs(start_y) if radius_y - abs(start_y) > 0 else 10000
                        ad_end_x = radius_x - abs(end_x) if radius_x - abs(end_x) > 0 else 10000
                        ad_end_y = radius_y - abs(end_y) if radius_y - abs(end_y) > 0 else 10000
                        label_min = min(ad_start_x, ad_start_y, ad_end_x, ad_end_y)
                        if ad_start_x == label_min or ad_start_y == label_min:
                            if ad_start_x == label_min:
                                tmp += 1
                                if start_x < 0:
                                    start_point = re_rotate([- radius_x - size, start_point[1]], cos, sin, [center_x, center_y])
                                else:
                                    start_point = re_rotate([radius_x + size, start_point[1]], cos, sin, [center_x, center_y])

                            elif ad_start_y == label_min:
                                tmp += 1
                                if start_y < 0:
                                    start_point = re_rotate([start_point[0], -radius_y - size], cos, sin, [center_x, center_y])

                                else:
                                    start_point = re_rotate([start_point[0], radius_y + size], cos, sin, [center_x, center_y])

                            return start_point, re_rotate(end_point, cos, sin, [center_x, center_y]), tmp
                        else:
                            if ad_end_x == label_min:
                                if end_x < 0:
                                    end_point = re_rotate([- radius_x - size, end_point[1]], cos, sin, [center_x, center_y])

                                else:
                                    end_point = re_rotate([radius_x + size, end_point[1]], cos, sin, [center_x, center_y])

                            elif ad_end_y == label_min:
                                if end_y < 0:
                                    end_point = re_rotate([end_point[0], -radius_y - size], cos, sin, [center_x, center_y])

                                else:
                                    end_point = re_rotate([end_point[0], radius_y + size], cos, sin, [center_x, center_y])

                            return re_rotate(start_point, cos, sin, [center_x, center_y]), end_point, tmp

                # 点在建筑内
                if Polygon(building).contains(Point(points[i])):
                    points[i] = adjust_inter_point(points[i], 1)
                    if i == 0:
                        points[-1] = points[0]
                if Polygon(building).contains(Point(points[i+1])):
                    points[i+1] = adjust_inter_point(points[i+1], 1)
                    if i+1 == len(points) - 1:
                        points[0] = points[-1]

                # 道路经过建筑
                line = LineString([points[i], points[i+1]])
                while line.intersects(Polygon(building)):
                    # ax = plt.gca()
                    # GeoSeries(Polygon(building)).plot(ax=ax, color='red')
                    # GeoSeries(line).plot(ax=ax, color='green')
                    # ax.set_aspect(1)
                    # # plt.axis('off')
                    # plt.show()
                    points[i], points[i+1], tmp = adjust_outer_point(i, points[i], points[i+1], 1)
                    if i == 0:
                        points[-1] = points[0]
                    if i+1 == len(points) - 1:
                        points[0] = points[-1]
                    line = LineString([points[i], points[i+1]])
                    # ax = plt.gca()
                    # GeoSeries(Polygon(building)).plot(ax=ax, color='red')
                    # GeoSeries(line).plot(ax=ax, color='green')
                    # ax.set_aspect(1)
                    # # plt.axis('off')
                    # plt.show()
                    label += tmp

                if label > 0:
                    print('again')
                    print(f'lable is : {label}')
                    print(f'length is: {points}')
                    self.data_post_process(points, buildings)
        return points

    @staticmethod
    def show_picture(i, base_shapes, buildings, generator_points):
        '''
            数据可视化
        :param base_shapes: array,地块数据
        :param buildings: np.array,建筑数据
        # :param init_points: np.array,道路数据
        :param generator_points: np.array,训练数据
        :return: picture
        '''

        # 地块
        base_shapes = Polygon(base_shapes)

        # 建筑
        building_shapes = []
        for building in buildings:
            building_shapes.append(Polygon(building))

        # 道路
        # init_points = LineString(reset_endpoints(init_points))
        generator_points = LineString(generator_points)

        # 可视化
        ax = plt.gca()
        GeoSeries(base_shapes).plot(ax=ax, color='blue')
        GeoSeries(building_shapes).plot(ax=ax, color='red')
        # GeoSeries(init_points).plot(ax=ax, color='green')
        GeoSeries(generator_points).plot(ax=ax, color='yellow')
        ax.set_aspect(1)
        plt.axis('off')
        plt.savefig(f'datasets/test/{str(i)}.jpg')
        plt.close()

    def output_data(self, test_url, testing=False):
        '''
            对预测数据进行规范化、预测、还原、展示
        :param test_url: str,数据地址
        :return:
        '''
        test_data, slope, inter = self.set_x(test_url)
        predict_data = self.predict(test_data)
        predict_paths = self.reset_data(predict_data, slope, inter)
        outputs = []
        if testing:
            line_data = self.set_y(test_url)
            init_paths = self.reset_data(line_data, slope, inter)
        for i in range(len(predict_paths)):
            endpoints = reset_endpoints(predict_paths[i])
            predict_path = self.data_post_process(endpoints, self.multi_buildings[i])
            outputs.append(predict_path)
            print('data post process finished!')
            self.show_picture(i, self.multi_bases[i], self.multi_buildings[i], predict_path)
        return outputs
if __name__ == '__main__':
    seq = Seq2seq()
    # seq.train(TRAIN_URL, 200)
    seq.output_data('datasets/test_one.txt')