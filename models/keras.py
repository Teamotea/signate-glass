import keras
from keras import Sequential
from keras.optimizers import Adam


class Keras(keras.Model):
    def __init__(self, name):
        super(Keras, self).__init__()
        pass

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass
