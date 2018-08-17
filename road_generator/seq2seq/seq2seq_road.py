import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import seq2seq
from seq2seq.models import Seq2Seq
from shapely.geometry import Polygon, LineString, MultiPoint
from matplotlib import pyplot as plt
from geopandas import GeoSeries
import cv2


EPOCHS = 2000
latent_dim = 256
data_url = 'line_plan.txt'

class Seq2seq():
    def __init__(self, epochs, latent_dim, url):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.data_url = url

        self.encoder_data, self.decoder_data = self.set_data(self.data_url)

        # seq2seq
        self.model = Seq2Seq(batch_input_shape=(None, self.max_encoder_length, 2), hidden_dim=256, output_dim=2,
                        output_length=self.max_decoder_length, depth=3)

        self.model.compile(loss='mse', loss_weights=[100], optimizer='rmsprop')

        self.model.fit(self.encoder_data, self.decoder_data, batch_size=1, epochs=EPOCHS)
        self.model.save_weights('s2s_1.h5')
        # self.model.load_weights('s2s_1.h5')

    def set_data(self, url):
        """
            对数据进行归一化
        :param url: 数据连接
        :return: data
        """
        inputs_data = []
        output_data = []
        with open(url, 'r') as plan:
            pl = plan.readlines()

            for line in pl:
                data = eval(line.strip('\n'))
                self.id = data['id']

                # 用于max_min归一化
                self.base_shapes = data['block']
                x_coords, y_coords = zip(*self.base_shapes)
                self.x_min = min(x_coords)
                self.x_max = max(x_coords)
                self.y_min = min(y_coords)
                self.y_max = max(y_coords)

                # # 对建筑进行中心化处理
                # self.buildings = data['buildings']
                # center_buildings = []
                # for building in self.buildings:
                #     x, y = zip(*building)
                #     center_buildings.append([np.mean(x), np.mean(y)])
                #
                # # 对建筑进行归一化处理
                # for center in center_buildings:
                #     center[0] = (center[0] - self.x_min) / (self.x_max - self.x_min)
                #     center[1] = (center[1] - self.y_min) / (self.y_max - self.x_min)

                # 对建筑进行外接 + 归一化处理
                self.buildings = data['buildings']
                self.normal_buildings = []
                self.boxs = []
                for building in self.buildings:
                    rect = cv2.minAreaRect(np.array(building, np.int32))
                    box = cv2.boxPoints(rect)
                    self.boxs.append(box)
                    for i in range(4):
                        x = (box[i][0] - self.x_min) / (self.x_max - self.x_min)
                        y = (box[i][1] - self.y_min) / (self.y_max - self.y_min)
                        self.normal_buildings.append([x, y])

                # 对道路进行归一化处理
                endpoints = []
                self.lines = data['lines']
                for path in self.lines:
                    x = (path[0] - self.x_min) / (self.x_max - self.x_min)
                    y = (path[1] - self.y_min) / (self.y_max - self.y_min)
                    endpoints.append([x, y])

                inputs_data.append(self.normal_buildings)

                output_data.append(endpoints)

        num = len(inputs_data)
        self.max_encoder_length = max([len(inputs) for inputs in inputs_data])
        self.max_decoder_length = max([len(outputs) for outputs in output_data])

        print('Number of data:', num)
        print('Max length for inputs:', self.max_encoder_length)
        print('Max length for outputs:', self.max_decoder_length)

        # 数据格式化处理
        encoder_data = np.zeros((num, self.max_encoder_length, 4, 2))
        decoder_data = np.zeros((num, self.max_decoder_length, 2))

        for i in range(num):
            for j in range(len(inputs_data[i])):
                encoder_data[i, j] = inputs_data[i][j]

            for k in range(len(output_data[i])):
                decoder_data[i, k] = output_data[i][k]
        return encoder_data, decoder_data

    def reset_data(self, datasets):
        '''
            对数据进行复原
        :param datasets: 待处理数据
        :return: data
        '''
        for data in datasets:
            for i in range(len(data)):
                if data[i][0] < 0:
                    data[i][:] = 0
                elif data[i][1] < 0:
                    data[i][:] = 0
                else:
                    data[i][0] = data[i][0] * (self.x_max - self.x_min) + self.x_min
                    data[i][1] = data[i][1] * (self.y_max - self.y_min) + self.y_min
        return datasets

    @staticmethod
    def show_picture(id, base_shapes, buildings, init_points):
        '''
            数据可视化，结果对比
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
        init_points = MultiPoint(init_points)
        # generator_points = MultiPoint(points)

        # 可视化
        ax = plt.gca()
        GeoSeries(base_shapes).plot(ax=ax, color='blue')
        GeoSeries(building_shapes).plot(ax=ax, color='red')
        GeoSeries(list(init_points)).plot(ax=ax, color='green')
        #GeoSeries(list(generator_points)).plot(ax=ax, color='yellow')
        ax.set_aspect(1)
        plt.axis('off')
        plt.show()

    def test(self):
        test_data = np.array([self.encoder_data[-1]])
        #predict_data = self.model.predict(test_data)
        #paths = self.reset_data(predict_data)[0]
        init_paths = self.lines
        self.show_picture(self.id, self.base_shapes, self.boxs, init_paths)

if __name__ == '__main__':
    seq = Seq2seq(EPOCHS, latent_dim, data_url)
