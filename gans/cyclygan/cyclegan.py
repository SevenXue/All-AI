from keras.models import Model, Sequential
from keras.layers import Dense, Input, Reshape, Dropout, Concatenate, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam
from keras_contrib.layers.normalization import InstanceNormalization
import time
import numpy as np
from data_loader import DataLoader
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model

class CycleGan():

    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # configure data loader
        self.dataset_name = 'apple2orange'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        # calulate output shape of D (patchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        #Loss weights
        self.lambda_cycle = 10.0
        self.lambda_id = 0.1 * self.lambda_cycle

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.discriminator()
        self.d_B = self.discriminator()
        self.d_A.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.d_B.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        # save d_model as picture
        plot_model(self.d_A, to_file='visual/d_A.png', show_shapes=True, show_layer_names=True)
        plot_model(self.d_B, to_file='visual/d_B.png', show_shapes=True, show_layer_names=True)

        #Build the generator
        self.g_AB = self.generator()
        self.g_BA = self.generator()

        # save g_model as picture
        plot_model(self.g_AB, to_file='visual/g_AB.png', show_shapes=True, show_layer_names=True)
        plot_model(self.g_BA, to_file='visual/g_BA.png', show_shapes=True, show_layer_names=True)

        #input images from both domain
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        # translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators detemines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])

        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

        # save combine_model as picture
        plot_model(self.combined, to_file='visual/combined.png', show_shapes=True, show_layer_names=True)


    def generator(self):
        """
            构造了生成器，使用U-net网络
        :return: generator
        """

        def conv2d(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)

            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):

            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # image input
        d0 = Input(shape=self.img_shape)

        #dowmsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        #upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        outut_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, outut_img)



    def discriminator(self):
        """
            构造一个判别器
        :return:
        """

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = time.time()

        valid = np.ones((batch_size,) + self.disc_patch)

        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # train discriminators

                # translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # train generators

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B])

                run_time = time.time() - start_time

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s"
                      % (epoch, epochs,
                         batch_i, self.data_loader.n_batches,
                         d_loss[0], 100*d_loss[1],
                         g_loss[0],
                         np.mean(g_loss[1:3]),
                         np.mean(g_loss[3:5]),
                         np.mean(g_loss[5:6]),
                         run_time))

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):

        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain='B', batch_size=1, is_testing=True)

        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt=0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig('images/%s/%d_%d.png' %(self.dataset_name, epoch, batch_i))
        plt.close()

if __name__ == '__main__':
    gan = CycleGan()
    gan.train(epochs=200, batch_size=1, sample_interval=200)









