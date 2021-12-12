from tensorflow.python.framework.ops import reset_default_graph
import tensorflow_datasets as tsdf
import tensorflow as tf
from tensorflow.keras import layers
from keras import regularizers
import matplotlib.pyplot as plt
import sys
import numpy as np
from augment import *

EPOCHS_BATCH = 2

def prepare(ds, batch_size, shuffle=False, augment=False):
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

tests = [
    {
        "dataset": "cifar10",
        "size_x": 32,
        "size_y": 32,
        "augment": True,
        "epochs": 250,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "split": ['train[:80%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "cifar10",
        "size_x": 32,
        "size_y": 32,
        "augment": False,
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "split": ['train[:80%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "cifar10",
        "size_x": 32,
        "size_y": 32,
        "augment": True,
        "epochs": 250,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "split": ['train[:80%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "cifar10",
        "size_x": 32,
        "size_y": 32,
        "augment": True,
        "epochs": 250,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "split": ['train[:80%]', 'train[80%:]', 'test']
    },
]

datasets = ['cifar10', 'cifar100']

res = open("results.txt", "w")
print("something", file=res, flush=True)

for test in tests:
    tf.random.set_seed(1234)
    init_seed(4321)
    print(test, file=res, flush=True)

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(test['size_x'], test['size_y']),
        layers.Rescaling(1./255)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        #layers.Lambda(lambda x: saturate(x, 0.25)),
        layers.Lambda(lambda x: shiney(x, 0.2)),
        #layers.Lambda(lambda x: crop(x, 0.9, test['size_x'], test['size_y']))
    ])

    (ds_train, ds_val, ds_test), meta = tsdf.load(
        test['dataset'],
        split=test['split'],
        data_dir=sys.path[0],
        with_info=True,
        as_supervised=True)

    num_classes = meta.features['label'].num_classes

    model = tf.keras.Sequential([
        layers.Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),

        layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(test['weight_decay']), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Flatten(),
        #layers.Dense(1024, activation='elu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=test['learning_rate']),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    ds_train = prepare(ds_train, batch_size=test['batch_size'], shuffle=True, augment=test['augment'])
    ds_val = prepare(ds_val, batch_size=test['batch_size'])
    ds_test = prepare(ds_test, batch_size=test['batch_size'])

    #for example in ds_train:
    #    img = np.array(example[0][0])
    #    plt.figure()
    #    plt.imshow(img)
    #    plt.show()

    ep = test['epochs']
    for inter in range(ep//EPOCHS_BATCH):
        model.fit(
            ds_train,
            validation_data = ds_val,
            epochs = EPOCHS_BATCH
        )

        _, acc_train = model.evaluate(ds_train)
        _, acc_test = model.evaluate(ds_test)
        _, acc_val = model.evaluate(ds_val)
        print("Accuracy after %d epochs: Train: %f, Val: %f, Test: %f" %(EPOCHS_BATCH*(inter+1), acc_train, acc_val, acc_test), 
            file=res, flush=True)