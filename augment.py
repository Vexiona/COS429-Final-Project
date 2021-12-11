import tensorflow as tf
from tensorflow.keras import layers
import random

def init_seed(seed):
    random.seed(seed)

def rand(x, y):
    return int(random.random()*(y-x)+x)

def gen_tf_seed():
    return (random.randint(1, 1000), random.randint(1, 1000))

def saturate(x):
    return tf.image.stateless_random_saturation(x, 0.5, 2, seed=gen_tf_seed())

def crop(x, size_x, size_y):
    return layers.Resizing(size_x, size_y)(
        tf.image.stateless_random_crop(x, [rand(0.8*size_x, size_x), rand(0.8*size_y, size_y), 3], seed=gen_tf_seed())
    )