from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.affinity import translate
from building.prototype import get_prototype
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imresize, imread
import random


def create_pic(url, save_dir):
     arrangement = eval(url.strip("\n"))
     base = Polygon(arrangement["base"])
     centr = arrangement["centroids"]
     pid = arrangement["pid"]
     shape = get_prototype(pid).shape
     shapes = [translate(shape, x, y) for x, y in centr]
     ax = plt.gca()
     GeoSeries([base]).plot(ax=ax, color='blue')
     GeoSeries(shapes).plot(ax=ax, color='red')
     ax.set_aspect(1)
     plt.axis('off')
     plt.savefig(save_dir)
     plt.close()

class DataLoader():

    def __init__(self, data_name, img_res=(128, 128)):
        self.data_name = data_name
        self.img_res = img_res


    def load_data(self, num, is_testing=False):
        init_path = glob('datasets/%s/init/*' % self.data_name)
        build_path = glob('datasets/%s/building/*' % self.data_name)

        imgs_A, imgs_B = [], []
        for i in range(num):
            img_a = self.img_read(build_path[i])
            img_b = self.img_read(init_path[i])

            img_a = imresize(img_a, self.img_res)
            img_b = imresize(img_b, self.img_res)

            if not is_testing and np.random.random() > 0.5:
                img_a = np.fliplr(img_a)
                img_b = np.fliplr(img_b)

            imgs_A.append(img_a)
            imgs_B.append(img_b)
        imgs_A = np.array(imgs_A) / 127.5 - 1
        imgs_B = np.array(imgs_B) / 127.5 - 1

        return imgs_A, imgs_B

    def load_data_batches(self, num):
        init_path = glob('datasets/%s/init/*' % self.data_name)
        build_path = glob('datasets/%s/building/*' % self.data_name)

        for i in range(num):
            imgs_A, imgs_B = [], []
            img_A = self.img_read(build_path[i])
            img_B = self.img_read(init_path[i])

            img_A = imresize(img_A, self.img_res)
            img_B = imresize(img_B, self.img_res)

            if np.random.random() > 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B

    def test_data(self, num, batch_size=1):
        init_path = glob('datasets/%s/init/*' % self.data_name)
        build_path = glob('datasets/%s/building/*' % self.data_name)

        self.n_batches = int(len(init_path))
        #print(self.n_batches)
        test_id = random.randint(num, self.n_batches-1)
        #print(build_path[test_id])

        imgs_A = []
        imgs_B = []
        img_A = self.img_read(build_path[test_id])
        img_B = self.img_read(init_path[test_id])

        img_A = imresize(img_A, self.img_res)
        img_B = imresize(img_B, self.img_res)

        imgs_A.append(img_A)
        imgs_B.append(img_B)

        imgs_A = np.array(imgs_A) / 127.5 - 1.
        imgs_B = np.array(imgs_B) / 127.5 - 1.

        return imgs_A, imgs_B

    def img_read(self, path):
        return imread(path, mode='RGB').astype(np.float)


if __name__ == '__main__':
    a = DataLoader('shenzhen_1')
    a.load_data(num=1)


