import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
import os

from scipy.misc import imsave as ims
from utils import *
from ops import *


class LatentAttention():
    def __init__(self, epoch=10):

        self.epoch = epoch
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500  # hidden node amount
        self.n_z = 20  # latent vector size
        self.batch_size = 256

        self.train_num = self.n_samples // self.batch_size

        self.images = tf.placeholder(tf.float32, [None, 784])  # input image
        image_matrix = tf.reshape(self.images, [-1, 28, 28, 1])  # input reshape [batch_size, h, w, c)

        # get z_mean, z_stddev from the image. remember it is just graph
        z_mean, z_stddev = self.recognition(image_matrix)

        # get latent_vector_z. remember it is just graph
        samples = tf.random_normal([self.batch_size, self.n_z], 0, 1, dtype=tf.float32)
        latent_vector_z = z_mean + (z_stddev * samples)

        # generate image from latent_vector_z
        self.generated_images = self.generation(latent_vector_z)
        generated_flat = tf.reshape(self.generated_images, [self.batch_size, 28 * 28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat)
                                              + (1-self.images) * tf.log(1e-8 + 1 - generated_flat), 1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev)
                                               - tf.log(tf.square(z_stddev)) - 1, 1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):

            # lrelu, conv2d - customized wrapper function for tf API
            # conv2d: 5x5 filter, 2x2 stride, padding=same(means same if stride is 1)
            # therefore (28 - 5 + 4) / 2 + 1 = 14
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2, [self.batch_size, 7 * 7 * 32])

            # wonder it's okay to use the same function twice
            # and how it manage weights. maybe get_Variables() function is key
            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batch_size, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batch_size, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batch_size, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        #visualization = self.mnist.train.next_batch(self.batch_size)[0]  # we do not need label (= [1])
        visualization = self.mnist.test.next_batch(self.batch_size)[0]  # we do not need label (= [1])
        reshaped_vis = visualization.reshape(self.batch_size, 28, 28)

        # get first 64 images and merge in to one image of 8 x 8 grid, and SAVE
        # for reference to compare generated images
        ims("results/base.jpg", merge(reshaped_vis[:64], [8, 8]))

        # train
        saver = tf.train.Saver(max_to_keep=2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epoch):
                for idx in range(self.train_num):
                    batch = self.mnist.train.next_batch(self.batch_size)[0]

                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                     feed_dict={self.images: batch})

                    # dumb hack to print cost every epoch
                    # if idx % (self.n_samples - 3) == 0: # original code. it works, but it seems wrong
                    if idx % (self.train_num - 3) == 0:
                        print("epoch %d: gen_loss %f lat_loss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train", global_step=epoch)

                        # generate test
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batch_size, 28, 28)
                        ims("results/"+str(epoch)+".jpg", merge(generated_test[:64], [8, 8]))


model = LatentAttention()
model.train()
