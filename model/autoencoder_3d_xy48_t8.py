# The 48*48*8 version of AE structures
# 3D version, for time sequence
# Liya Ding
# 2024.09

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv3D, Flatten, Dense, Conv3DTranspose, Reshape


def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(8, 48, 48, 1), name='layers_flatten'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
    tf.keras.layers.Dropout(0.2, name='layers_dropout'),
    tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
  ])

class AE(tf.keras.Model):
    def __init__(self, latent_dim: int, net_type:str='simple'):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        assert net_type in ['simple', 'conv']
        if net_type == "simple":
            self.inference_net = tf.keras.Sequential([
                InputLayer(input_shape=[8, 48, 48]),
                Flatten(),
                Dense(4096, activation='relu'),
                Dense(1024, activation='relu'),
                Dense(512, activation='relu'),                
                Dense(128, activation='relu'),
                Dense(self.latent_dim),
            ])
            self.generative_net = tf.keras.Sequential([
                InputLayer(input_shape=[self.latent_dim]),
                Dense(128, activation='relu'),
                Dense(512, activation='relu'),
                Dense(1024, activation='relu'),
                Dense(4096, activation='relu'),
                Dense(48 *48 * 8 * 1, activation='sigmoid'),
                Reshape(target_shape=[8, 48, 48]),
            ])
        if net_type == "conv":
            self.inference_net = tf.keras.Sequential([
                InputLayer(input_shape=[8, 48, 48, 1]),
                Conv3D(
                    filters=8, kernel_size=3, strides=(2, 2, 2),  padding="SAME",activation='relu'),
                Conv3D(
                    filters=16, kernel_size=[3,3,3], strides=(2, 2, 2),  padding="SAME",activation='relu'),
                Conv3D(
                    filters=32, kernel_size=[1,3,3], strides=(2, 2, 2),  padding="SAME",activation='relu'),
                Flatten(),
                Dense(128, activation='relu'),
                # No activation
                Dense(self.latent_dim),
            ])
            self.generative_net = tf.keras.Sequential([
                InputLayer(input_shape=[self.latent_dim]),
                Dense(128, activation='relu'),
                Dense(1* 6 * 6 * 32, activation='relu'),
                Reshape(target_shape=(1, 6, 6, 32)),
                Conv3DTranspose(
                    filters=16,
                    kernel_size=[1,3,3],
                    strides=(2, 2, 2),
                    padding="SAME",
                    activation='relu'),
                Conv3DTranspose(
                    filters=8,
                    kernel_size=[3,3,3],
                    strides=(2, 2, 2),
                    padding="SAME",
                    activation='relu'),
                Conv3DTranspose(
                    filters=1,
                    kernel_size=3,
                    strides=(2, 2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                Conv3DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1, 1), padding="SAME", activation='sigmoid'),
            ])
            
    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z):
        logits = self.generative_net(z)
        return logits


