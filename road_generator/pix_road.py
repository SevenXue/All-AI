from keras.models import Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Dropout, Input, Concatenate
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from load_data import DataLoader
from ann_visualizer.visualize import ann_viz
from keras.callbacks import TensorBoard


class PixRoad():

    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.data_name = 'shenzhen_1'
        self.data_loader = DataLoader(data_name=self.data_name,
                                      img_res=(self.img_cols, self.img_rows))

        # create package of view
        os.makedirs('view', exist_ok=True)

        # number of first layer of G and D
        self.gf = 64
        self.df = 64

        # output shape of D
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        optimizer = Adam(0.0002, 0.5)

        # build and complie the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        plot_model(self.discriminator, to_file='view/discriminator.png', show_shapes=True)
        #ann_viz(self.discriminator, view=True, filename='view/discriminator.gv', title='Discriminator')

        # build the generator
        self.generator = self.bulid_generator()
        plot_model(self.generator, to_file='view/generator.png', show_shapes=True)

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_A = self.generator(img_B)

        # just train the generator
        self.discriminator.trainable = False

        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        plot_model(self.combined, to_file='view/combined.png', show_shapes=True)


    def bulid_generator(self):
        """U-net Generator"""

        def conv(input_layer, filters, f_size=4, bn=True):

            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(input_layer)
            d = LeakyReLU()(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)

            return d

        def deconv(input_layer, skip_input, filters, f_size=4, dropout_rate=0):

            u = UpSampling2D(size=2)(input_layer)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])

            return u

        d0 = Input(shape=self.img_shape)

        #downsampling
        d1 = conv(d0, self.gf, bn=False)
        d2 = conv(d1, self.gf*2)
        d3 = conv(d2, self.gf*4)
        d4 = conv(d3, self.gf*8)
        d5 = conv(d4, self.gf*8)

        #upsampling
        u1 = deconv(d5, d4, self.gf*8)
        u2 = deconv(u1, d3, self.gf*4)
        u3 = deconv(u2, d2, self.gf*2)
        u4 = deconv(u3, d1, self.gf)

        u5 = UpSampling2D(size=2)(u4)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)

        return Model(d0, output_img)

    def build_discriminator(self):

        def conv2(input_layer, filters, f_size=4, bn=True):

            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(input_layer)
            d = LeakyReLU()(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        conbined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = conv2(conbined_imgs, self.df, bn=False)
        d2 = conv2(d1, self.df*2)
        d3 = conv2(d2, self.df*4)
        d4 = conv2(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train_batches(self, epochs, num, batch_size=600, sample_interval=50):
        start_time = time.time()

        valid = np.ones((batch_size,) + self.disc_patch)
        print(valid.shape)
        fake = np.zeros((batch_size,) + self.disc_patch)
        print(fake.shape)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_data_batches(num)):
                fake_A = self.generator.predict(imgs_B)

                # train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train then generator
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])


                run_time = time.time() - start_time

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" %
                      (epoch, epochs,
                       batch_i, num,
                       d_loss[0], 100*d_loss[1],
                       g_loss[0],
                       run_time))
                if batch_i % sample_interval == 0:
                    self.sample_image(epoch, batch_i)

    def train(self, epochs, batch_size=600):

        # create labels
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            print('This is {} times'.format(epoch))
            imgs_A, imgs_B = self.data_loader.load_data(600)

            fake_A = self.generator.predict(imgs_B)

            # train the discriminator
            self.discriminator.fit([imgs_A, imgs_B], valid, batch_size=10, shuffle=False)
            self.discriminator.fit([fake_A, imgs_B], fake, batch_size=10, shuffle=False)

            # train the generator
            os.makedirs('tensorboard/combined', exist_ok=True)
            self.combined.fit([imgs_A, imgs_B], [valid, imgs_A],
                              callbacks=[TensorBoard(log_dir='tensorboard/combined', write_images=True, histogram_freq=1)],
                              shuffle=False)

    def sample_image(self, epoch, batch_i):
        os.makedirs('images/%s' % self.data_name, exist_ok=True)

        imgs_A, imgs_B = self.data_loader.test_data(800)
        # imgs_B = None
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, imgs_A, fake_A])

        gen_imgs = 0.5 * gen_imgs + 0.5

        #print(len(gen_imgs))

        titles = ['Conditions', 'Original', 'Generated']

        fig, axs = plt.subplots(3, 1)
        cnt = 0
        for i in range(3):
            axs[i].imshow(gen_imgs[cnt])
            axs[i].set_title(titles[cnt])
            axs[i].axis('off')
            cnt += 1
        fig.savefig('images/%s/%d_%d.png' % (self.data_name, epoch, batch_i))
        plt.close()

if __name__ == '__main__':
    gan = PixRoad()
    gan.train(epochs=3, batch_size=600)



