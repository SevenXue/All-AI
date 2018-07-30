import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imresize, imread
import random

class DataLoader():

    def __init__(self, data_name, img_res=(256, 256)):
        self.data_name = data_name
        self.img_res = img_res

    @staticmethod
    def resize_picture(url, shapes):
        img = imread(url, mode='RGB').astype(np.float)
        img = imread(img, shapes)
        return img


    def load_data(self, urls, num=0):
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

            if np.random.random() > 0.5:
                img_a = np.fliplr(img_a)
                img_b = np.fliplr(img_b)

            imgs_A.append(img_a)
            imgs_B.append(img_b)

        # 规范化数据
        imgs_A = np.array(imgs_A) / 127.5 - 1
        imgs_B = np.array(imgs_B) / 127.5 - 1

        return imgs_A, imgs_B

if __name__ == '__main__':
    a = DataLoader('shenzhen_1')
    a.load_data(num=1)
