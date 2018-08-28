import numpy as np
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from seq2seq.models import AttentionSeq2Seq
from shapely.geometry import Polygon, LineString
from matplotlib import pyplot as plt
from geopandas import GeoSeries
import cv2
import os

os.makedirs('model', exist_ok=True)
os.makedirs('tensorboard', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

EPOCHS = 500
latent_dim = 256
data_url = 'datasets/train_data.txt'
model_file = 'model/s2s_rich_256.h5'

class Seq2seq():

    def __init__(self, epochs, latent_dim):
        # self.epochs = epochs
        self.latent_dim = latent_dim
        self.max_encoder_length = 49
        self.max_decoder_length = 17

        # seq2seq
        self.model = AttentionSeq2Seq(batch_input_shape=(None, self.max_encoder_length, 2), hidden_dim=self.latent_dim, output_dim=2,
                                      output_length=self.max_decoder_length, depth=3)

        self.model.compile(loss='mse', loss_weights=[200], optimizer='rmsprop')

        if os.path.exists(model_file):
            self.model.load_weights(model_file)
        else:
            print('Model needs to train.')

    def train(self):
        if os.path.exists(data_url):
            encoder_data, slope, inter = self.set_x(data_url)
            decoder_data = self.set_y(data_url)
            self.model.fit(encoder_data, decoder_data, batch_size=8, epochs=EPOCHS,
                           callbacks=[TensorBoard(log_dir='tensorboard/seq_5')])
            self.model.save_weights(model_file)
        else:
            raise ValueError('data is not existing.')

    @staticmethod
    def sort_paths(endpoints):
        '''
            对点进行逆时针排序
        :param endpoints:
        :return:
        '''
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
            对x进行预处理
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
            对y进行预处理
        :param url: str,
        :return: np.array
        '''
        output_data = []
        with open(url, 'r') as plan:
            pl = plan.readlines()

            for line in pl:
                data = eval(line.strip('\n'))
                # self.id = data['id']

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

    def predict(self, pre_data):
        '''
            预测
        :param pre_data:
        :return:
        '''
        self.model.load_weights(model_file)
        result = self.model.predict(pre_data)
        return result



    @staticmethod
    def show_picture(id, base_shapes, buildings, init_points, generator_points):
        '''
            数据可视化
        :param id:
        :param base_shapes: 地块数据
        :param buildings: 建筑数据
        :param road_line: 道路数据
        :return: picture
        '''

        # 地块
        base_shapes = Polygon(base_shapes)

        # 建筑
        building_shapes = []
        for building in buildings:
            building_shapes.append(Polygon(building))


        # 道路
        def reset_endpoints(points):
            endpoints = []
            for i in range(len(points)):
                if points[i][0] != 0:
                    endpoints.append(points[i])
            if len(endpoints) != 0:
                endpoints.append(points[0])
            return endpoints

        init_points = LineString(reset_endpoints(init_points))
        generator_points = LineString(reset_endpoints(generator_points))

        # 可视化
        ax = plt.gca()
        GeoSeries(base_shapes).plot(ax=ax, color='blue')
        GeoSeries(building_shapes).plot(ax=ax, color='red')
        GeoSeries(init_points).plot(ax=ax, color='green')
        GeoSeries(generator_points).plot(ax=ax, color='yellow')
        ax.set_aspect(1)
        plt.axis('off')
        plt.show()

    def test(self, test_url):
        test_data, slope, inter = self.set_x(test_url)
        line_data = self.set_y(test_url)
        predict_data = self.predict(test_data)
        predict_paths = self.reset_data(predict_data, slope, inter)
        init_paths = self.reset_data(line_data, slope, inter)
        for i in range(len(predict_paths)):
            self.show_picture(self.id, self.multi_bases[i], self.multi_buildings[i], init_paths[i], predict_paths[i])

if __name__ == '__main__':
    seq = Seq2seq(EPOCHS, latent_dim)
    seq.test('datasets/test_data.txt')