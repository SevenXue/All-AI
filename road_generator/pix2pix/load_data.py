#！ 数据处理和载入

import numpy as np
from glob import glob
from scipy.misc import imresize, imread

class DataLoader():
    """
        数据处理
    """

    def __init__(self, data_name, img_res=(256, 256)):
        self.data_name = data_name
        self.img_res = img_res

    @staticmethod
    def resize_picture(url, shapes):
        img = imread(url, mode='RGB').astype(np.float)
        img = imresize(img, shapes)
        return img

    def load_batch(self, num):
        path = glob('datasets/{}/buildings_train/*'.format(self.data_name))

        for i in range(num):
            imgs_A, imgs_B = [], []

            img_a = self.resize_picture(path[i], self.img_res)
            img_b = self.resize_picture(path[i].replace('buildings_train', 'init'), self.img_res)

            if np.random.random() > 0.5:
                img_a = np.fliplr(img_a)
                img_b = np.fliplr(img_b)

            imgs_A.append(img_a)
            imgs_B.append(img_b)

            imgs_A = np.array(imgs_A) / 127.5 - 1
            imgs_B = np.array(imgs_B) / 127.5 - 1

            yield imgs_A, imgs_B

    def load_data(self, urls, num=1):
        '''
            create the datasets
        :param num: the batch of datasets
        :param urls: local url of datasets
        :return: datasets
        '''

        imgs_A, imgs_B = [], []
        num = num if num else len(urls)
        for i in range(num):
            road_url = urls[i]
            init_url = urls[i].replace('buildings_train', 'init')
            init_url = init_url.replace('buildings_test', 'init')

            img_a = self.resize_picture(road_url, self.img_res)
            img_b = self.resize_picture(init_url, self.img_res)

            imgs_A.append(img_a)
            imgs_B.append(img_b)

        # 规范化数据
        imgs_A = np.array(imgs_A) / 127.5 - 1
        imgs_B = np.array(imgs_B) / 127.5 - 1

        return imgs_A, imgs_B

if __name__ == '__main__':
    a = DataLoader('shenzhen_1')

