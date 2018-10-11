from keras.models import Model
from keras.layers import Input, Reshape, Conv2D, Dense, Dropout, Flatten, BatchNormalization, Concatenate
import numpy as np
from keras.layers.noise import GaussianNoise
import matplotlib.pyplot as plt
from geopandas import GeoSeries
from shapely.geometry import Polygon, LineString
from data_process import sort_paths
import os

DIR = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
DATA_DIR = DIR + "/data/"
MODEL_DIR = DIR + "/model/"

class BuildingClassifier:
    def __init__(self, dim, epochs, batch_size, is_deep):
        self.dim = dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_deep = is_deep

        if self.is_deep:
            self.model = self.construct_deep_model()
            self.model_name = 'deep_classifier.h5'
        else:
            self.classify = self.construct_model()
            self.model_name = 'classifier.h5'
        self.model.summary()
        if os.path.exists(MODEL_DIR + self.model_name):
            self.model.load_weights(MODEL_DIR + self.model_name)
        else:
            print('Model need to be trained!')

    def construct_model(self):
        """
            structure a sample classifier for buildings
        :return:model of the classifier
        """
        # input&con
        input0 = Input(shape=(3, self.dim, self.dim), name='shapes')
        input1 = Reshape(target_shape=(self.dim, self.dim, 3))(input0)
        input2 = GaussianNoise(0.05)(input1)
        con1 = Conv2D(filters=4, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input2)
        drop1 = Dropout(0.5)(con1)
        bn1 = BatchNormalization(momentum=0.8)(drop1)
        con2 = Conv2D(filters=8, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn1)
        drop2 = Dropout(0.5)(con2)
        bn2 = BatchNormalization(momentum=0.8)(drop2)
        con3 = Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn2)
        drop3 = Dropout(0.5)(con3)
        bn3 = BatchNormalization(momentum=0.8)(drop3)
        con4 = Conv2D(filters=8, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn3)
        bn4 = BatchNormalization(momentum=0.8)(con4)
        input3 = Flatten()(bn4)

        # distances
        input4 = Input(shape=(1,), name='distances')
        input5 = Concatenate(axis=1)([input3, input4])
        dense2 = Dense(units=16, activation='relu')(input5)

        # output
        y = Dense(units=1, activation='sigmoid', name='styles')(dense2)
        model = Model(inputs=[input0, input4], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def construct_deep_model(self):
        """
            structure a classifier of buildings with deeper networks and more parameters.
        :return: the model of classifier
        """
        # input&con
        input0 = Input(shape=(3, self.dim, self.dim), name='shapes')
        input1 = Reshape(target_shape=(self.dim, self.dim, 3))(input0)
        con1 = Conv2D(filters=4, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input1)
        drop1 = Dropout(0.3)(con1)
        bn1 = BatchNormalization(momentum=0.8)(drop1)
        con2 = Conv2D(filters=8, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn1)
        drop2 = Dropout(0.3)(con2)
        bn2 = BatchNormalization(momentum=0.8)(drop2)
        con3 = Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn2)
        drop3 = Dropout(0.3)(con3)
        bn3 = BatchNormalization(momentum=0.8)(drop3)
        con4 = Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn3)
        bn4 = BatchNormalization(momentum=0.8)(con4)
        input3 = Flatten()(bn4)

        # distances
        input4 = Input(shape=(1,), name='distances')
        input5 = Concatenate(axis=1)([input3, input4])
        drop5 = Dropout(0.3)(input5)
        dense1 = Dense(units=32, activation='relu')(drop5)
        drop6 = Dropout(0.3)(dense1)
        dense2 = Dense(units=16, activation='relu')(drop6)
        drop7 = Dropout(0.3)(dense2)

        # output
        y = Dense(units=1, activation='sigmoid', name='styles')(drop7)
        model = Model(inputs=[input0, input4], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, data, save_model=False):
        """
        :param data: dict, data including shapes, styles and distances
        :param save_model: bool, save model or not
        """
        shapes = data["shapes"]
        styles = data["styles"]
        distances = data['distances']

        # split data
        print("split data")
        n = styles.shape[0]
        t = int(n * 0.9)
        training_shapes = shapes[:t]
        training_styles = styles[:t]
        training_distances = distances[:t]
        testing_shapes = shapes[t:]
        testing_styles = styles[t:]
        testing_distances = distances[t:]

        print("train model")
        self.model.fit(
            x=[training_shapes, training_distances],
            y=training_styles,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(
                {'shapes': testing_shapes, 'distances': testing_distances},
                {'styles': testing_styles}
            )
        )

        if save_model:
            print("save model")
            self.model.save_weights(MODEL_DIR + self.model_name)

    def predict(self, data):
        """
        :param data:dict, including shapes and distance
        :return: ndarray, (n, 1), labels of n buildings
        """
        shapes = data['shapes']
        distances = data['distances']
        if not os.path.exists(MODEL_DIR + self.model_name):
            print(MODEL_DIR+self.model_name)
            raise ValueError('Model need to be trained or the name of model is not right.')
        labels = self.model.predict(x=[shapes, distances])
        return labels

    @staticmethod
    def load_data(npz_file):
        """
        Load numpy arrays from files
        :param npz_file: str, url of npz_file
        :return: data: dict, data including shapes, styles and distances
        """
        data = np.load(npz_file)
        return data

def visual_data(data_file, labels, save=False):
    """
        visual data for more details
    :param data_file: str, url of data
    :param labels:ndarray, (n, 1), labels of n buildings
    :param save: bool, save data as image or not
    """
    with open(data_file, 'r') as df:
        lines = df.readlines()
        index = 0
        for line in lines:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            plt.axis('off')
            dataset = eval(line)
            data_id = dataset['id']
            block_shape = Polygon(dataset['block'])
            GeoSeries(block_shape).plot(ax=ax, color='blue')
            path = dataset['lines']
            path = sort_paths(path)
            path.append(path[0])
            path_shape = LineString(path)
            GeoSeries(path_shape).plot(ax=ax, color='green')
            buildings = dataset['buildings']
            for i in range(len(buildings)):
                building = Polygon(buildings[i])
                GeoSeries(building).plot(ax=ax, color='red' if labels[index][0] >= 0.5 else 'yellow')
                index += 1
            if save:
                os.makedirs('data/classifier', exist_ok=True)
                plt.savefig(f'data/classifier/{data_id}.jpg')
            else:
                plt.show()
            plt.close()

def test(data_file, npz_file, is_predicting=True):
    """
        load data, initial model and use model
    :param data_file: str, url for data
    :param npz_file: str url for stored npz file
    :param is_predicting: bool
    """
    if os.path.exists(npz_file):
        print('load data!')
        data = BuildingClassifier.load_data(npz_file)
    else:
        raise ValueError('prepare data firstly!')
    building_classifier = BuildingClassifier(256, 30, 16, True)
    if is_predicting:
        labels = building_classifier.predict(data)
        visual_data(data_file, labels)
    else:
        building_classifier.train(data)

if __name__ == '__main__':
    test('data/test_data.txt', 'data/test_data.npz', is_predicting=True)

