import tensorflow as tf
from tensorflow.keras import layers
import random

def init_seed(seed):
    random.seed(seed)

def rand(x, y):
    return random.random()*(y-x)+x

def gen_tf_seed():
    return (random.randint(1, 1000), random.randint(1, 1000))

@tf.function
def crop(x, factor, size_x, size_y):
    r = rand(factor, 1)
    return layers.Resizing(size_x, size_y)(
        tf.image.stateless_random_crop(x, [int(r*size_x), int(r*size_y), 3], seed=gen_tf_seed())
    )

@tf.function
def shiney(x, factor):
    return tf.image.stateless_random_brightness(x, factor, seed=gen_tf_seed())
@tf.function
def contrast(x, factor):
    return tf.image.stateless_random_contrast(x, 1-factor, 1, seed=gen_tf_seed())
@tf.function
def saturate(x, factor):
    return tf.image.stateless_random_saturation(x, 1-factor, 1+factor, seed=gen_tf_seed())
@tf.function
def hue(x, factor):
    return tf.image.stateless_random_hue(x, factor, seed=gen_tf_seed())

@tf.function
def invert(x, factor):
    random = tf.random.uniform(shape=[], minval=0, maxval=1)
    print(random)
    if random < factor:
        return tf.math.add(tf.math.multiply(x, -1), 1)
    return x
