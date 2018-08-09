from keras.optimizers import Adam
from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
import time
import numpy as np
from data_loader import DataLoader
import os
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz

class Pix2pix():

    def __init__(self):

        # input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.data_name,
                                    img_res=(self.img_rows, self.img_cols))

        #calulate output shape of D
        patch = int(self.img_rows / 2**4)
        self.disc_path = (patch, patch, 1)

        # number of filter in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        #build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        plot_model(self.discriminator, to_file='view/discriminator.png', show_shapes=True, show_layer_names=True)

        #ann_viz(self.discriminator, view=True, filename='view/discriminator.gv', title='discriminator')

        #build the generator
        self.generator = self.build_generator()
        plot_model(self.generator, to_file='view/generaor.png', show_shapes=True, show_layer_names=True)

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_A = self.generator(img_B)

        # for the combined model we just train the generator
        self.discriminator.trainable = False

        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        plot_model(self.combined, to_file='view/combined.png', show_shapes=True, show_layer_names=True)


    def build_generator(self):

        def conv(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU()(d)
            if bn:
                d = BatchNormalization()(d)
            return d

        def deconv(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
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
        d6 = conv(d5, self.gf*8)
        d7 = conv(d6, self.gf*8)

        #upsampling
        u1 = deconv(d7, d6, self.gf*8)
        u2 = deconv(u1, d5, self.gf*8)
        u3 = deconv(u2, d4, self.gf*8)
        u4 = deconv(u3, d3, self.gf*4)
        u5 = deconv(u4, d2, self.gf*2)
        u6 = deconv(u5, d1, self.df)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)




    def build_discriminator(self):

        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU()(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = conv2d(combined_imgs, self.df, bn=False)
        d2 = conv2d(d1, self.df*2)
        d3 = conv2d(d2, self.df*4)
        d4 = conv2d(d3, self.df*8)


        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = time.time()

        # adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_path)
        fake = np.zeros((batch_size,) + self.disc_path)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                fake_A = self.generator.predict(imgs_B)

                # train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train the generator
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                run_time = time.time() - start_time

                # plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time :%s" %
                      (epoch, epochs,
                      batch_i, self.data_loader.n_batches,
                      d_loss[0], 100*d_loss[1],
                      g_loss[0],
                      run_time))

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.data_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        gen_imgs = 0.5 * gen_imgs + 0.5

        print(len(gen_imgs))

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig('images/%s/%d_%d.png' % (self.data_name, epoch, batch_i))
        plt.close()

if __name__ == '__main__':
    gan = Pix2pix()
    #gan.train(epochs=100, batch_size=1, sample_interval=200)
