from keras.models import Model
from keras.layers import Input, Reshape, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, Concatenate
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
#from data_process import prepare_data
import os
import numpy as np
from keras import regularizers
from keras.layers.noise import GaussianNoise
from keras import optimizers
import matplotlib.pyplot as plt
from geopandas import GeoSeries
from seq2seq_road import sort_paths
import os
from data_process import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
class BuildingClassify:
    def __init__(self, dim, epochs, batch_size, model_file):
        self.dim = dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_file = model_file

        self.classify = self.classifier()
        self.classify.summary()
        if os.path.exists(model_file):
            self.classify.load_weights(model_file)
        else:
            print('Model need to be trained!')
        plot_model(self.classify, to_file='view/classify_16.png', show_shapes=True)

    def classifier(self):
        # input&con
        input0 = Input(shape=(3, self.dim, self.dim), name='shapes')
        input1 = Reshape(target_shape=(self.dim, self.dim, 3))(input0)
        input2 = GaussianNoise(0.05)(input1)
        con1 = Conv2D(filters=4, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input2)
        drop1 = Dropout(0.35)(con1)
        bn1 = BatchNormalization(momentum=0.8)(drop1)
        con2 = Conv2D(filters=8, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn1)
        drop2 = Dropout(0.35)(con2)
        bn2 = BatchNormalization(momentum=0.8)(drop2)
        con3 = Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn2)
        drop3 = Dropout(0.35)(con3)
        bn3 = BatchNormalization(momentum=0.8)(drop3)
        con4 = Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(bn3)
        # drop4 = Dropout(0.5)(con4)
        bn4 = BatchNormalization(momentum=0.8)(con4)
        input3 = Flatten()(bn4)

        # distances
        input4 = Input(shape=(1,), name='distances')
        input5 = Concatenate(axis=1)([input3, input4])
        drop5 = Dropout(0.3)(input5)
        dense2 = Dense(units=16, activation='relu')(drop5)
        drop6 = Dropout(0.0)(dense2)

        # output
        y = Dense(units=1, activation='sigmoid', name='styles')(drop6)
        model = Model(inputs=[input0, input4], outputs=[y])
        optimize = optimizers.SGD(lr=0.01)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    def train(self, data, model_file=None):
        shapes = data["shapes"]
        styles = data["styles"]
        distances = data['distances']

        print("split data")

        # split data
        n = styles.shape[0]
        t = int(n * 0.9)
        training_shapes = shapes[:t]
        training_styles = styles[:t]
        training_distances = distances[:t]
        testing_shapes = shapes[t:]
        testing_styles = styles[t:]
        testing_distances = distances[t:]
        print("train model")
        tensor_file = 'tensorboard/classify_1'
        os.makedirs(tensor_file, exist_ok=True)
        self.classify.fit(
            x=[training_shapes, training_distances],
            y=training_styles,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(
                {'shapes': testing_shapes, 'distances': testing_distances},
                {'styles': testing_styles}
            ),
            callbacks=[TensorBoard(log_dir=tensor_file, histogram_freq=2)] 
                       #EarlyStopping(monitor='loss', patience=3)],
        )

        if model_file is not None:
            print("save model")
            self.classify.save_weights(model_file)

    def predict(self, data):
        shapes = data['shapes']
        distances = data['distances']
        if not os.path.exists(self.model_file):
            raise ValueError('Model need to be trained or model file name is not right.')
        labels = self.classify.predict(x=[shapes, distances])
        print(labels.shape)
        return labels

    @staticmethod
    def load_data(npz_file):
        """
        Load numpy arrays from files
        :param npz_file:
        :return:
        """
        data = np.load(npz_file)
        return data
def visual_data_as_img(data_file, labels, save=True):
    """

    :param data_file:
    :param labels:
    :param save:
    :return:
    """
    with open(data_file, 'r') as df:
        os.makedirs('datasets/classify_16', exist_ok=True)
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
                plt.savefig(f'datasets/classify_16/{data_id}.jpg')
            else:
                plt.show()
            plt.close()

def get_classify_model(data_file, npz_file, is_testing=False):
    model_file = 'model/classify_mp_maxmin_16.h5'
    if os.path.exists(npz_file):
        print('load data!')
        data = BuildingClassify.load_data(npz_file)
    else:
        raise ValueError('prepare data!')
    building_classify = BuildingClassify(256, 30, 16, model_file)
    if is_testing:
        labels = building_classify.predict(data)
        visual_data_as_img(data_file, labels)
    else:
        building_classify.train(data, model_file=model_file)

if __name__ == '__main__':
    get_classify_model('datasets/test_data.txt', 'datasets/test_data.npz', is_testing=True)

