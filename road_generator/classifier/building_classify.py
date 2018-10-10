from keras.models import Model
from keras.layers import Input, Reshape, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, Concatenate
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoSeries
import os
from keras import regularizers
from keras.layers.noise import GaussianNoise

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
        plot_model(self.classify, to_file='view/classify.png', show_shapes=True)

    def classifier(self):

        # input&con
        input1 = Input(shape=(3, self.dim, self.dim), name='shapes')
        input2 = Reshape(target_shape=(self.dim, self.dim, 3))(input1)
        con1 = Conv2D(filters=4, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input2)
        bn1 = BatchNormalization(momentum=0.8)(con1)
        mp1 = MaxPool2D(pool_size=(2, 2))(bn1)
        con2 = Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(mp1)
        bn2 = BatchNormalization(momentum=0.8)(con2)
        mp2 = MaxPool2D(pool_size=(2, 2))(bn2)
        input3 = Flatten()(mp2)

        # dropout&dense
        drop1 = Dropout(0.2)(input3)
        dense1 = Dense(units=16, activation='relu')(drop1)
        drop2 = Dropout(0.5)(dense1)
        dense2 = Dense(units=8, activation='relu')(drop2)

        # input distances
        input4 = Input(shape=(1,), name='distances')
        input5 = Concatenate(axis=1)([dense2, input4])
        y = Dense(units=1, activation='sigmoid', name='styles')(input5)

        model = Model(inputs=[input1, input4], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def train(self, data, model_file=None):
        shapes = data["shapes"]
        styles = data["styles"]
        distances = data['distances']

        print("split and shuffled data")

        # split data
        n = styles.shape[0]
        t = int(n * 0.8)
        training_shapes = shapes[:t]
        training_styles = styles[:t]
        training_distances = distances[:t]
        testing_shapes = shapes[t:]
        testing_styles = styles[t:]
        testing_distances = distances[t:]
        print("train model")
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
            callbacks=[TensorBoard(log_dir='tensorboard/classify', histogram_freq=2)],
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

def visual_data_as_img(data_file, labels, save=False):
    with open(data_file, 'r') as df:
        os.makedirs('datasets/classify', exist_ok=True)
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
                # print(labels[index][0])
                index += 1
            if save:
                plt.savefig(f'datasets/classify/{data_id}.jpg')
            else:
                plt.show()
            plt.close()

def get_classify_model(data_file, npz_file, is_testing=False):
    model_file = 'model/classify_8.h5'
    if os.path.exists(npz_file):
        print('load data!')
        data = BuildingClassify.load_data(npz_file)
    elif os.path.exists(data_file):
        print('prepare data!')
        data = prepare_data(data_file, npz_file)
    building_classify = BuildingClassify(256, 30, 16, model_file)
    # if is_testing:
    #     labels = building_classify.predict(data)
    #     visual_data_as_img(data_file, labels)
    # else:
    #     building_classify.train(data, model_file=model_file)

if __name__ == '__main__':
    get_classify_model('datasets/test_data.txt', 'datasets/test_data.npz', is_testing=True)

