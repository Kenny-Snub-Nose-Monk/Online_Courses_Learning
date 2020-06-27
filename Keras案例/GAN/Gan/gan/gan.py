from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002)

        # 构建判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # 构建生成器
        self.generator = self.build_generator()

        # 生成器输入：噪音数据，输入：生成的图像数据
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # 在combined中只训练生成器
        self.discriminator.trainable = False

        # 最后由判别器来判断真假
        validity = self.discriminator(img)

        # 训练生成器去骗过判别器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, iter, batch_size=128, sample_interval=50):

        # 加载数据集
        (X_train, _), (_, _) = mnist.load_data()
        print('X_train.shape:',X_train.shape)
        # 数据预处理
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        print('X_train.shape:',X_train.shape)
        # 制作标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for i in range(iter):

            #训练判别器
            #训练一个batch数据
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 获取生成数据
            gen_imgs = self.generator.predict(noise)

            # 训练判别器
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 让生成器骗过判别器
            g_loss = self.combined.train_on_batch(noise, valid)

            # 打印训练结果
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss))

            # 保存部分迭代效果
            if i % sample_interval == 0:
                self.sample_images(i)

    def sample_images(self, iter):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # 预处理还原
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % iter)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(iter=50000, batch_size=32, sample_interval=500)
