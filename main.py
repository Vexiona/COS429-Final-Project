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
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "none",
        "epochs": 32,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:20%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "none",
        "epochs": 32,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:40%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "none",
        "epochs": 32,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:80%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "shallow",
        "epochs": 128,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:20%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "shallow",
        "epochs": 128,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:40%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "shallow",
        "epochs": 128,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:80%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "deep",
        "epochs": 256,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:20%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "deep",
        "epochs": 256,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:40%]', 'train[80%:]', 'test']
    },
    {
        "dataset": "svhn_cropped",
        "size_x": None,
        "size_y": None,
        "augment": "deep",
        "epochs": 256,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "split": ['train[:80%]', 'train[80%:]', 'test']
    },
]

res = open("results.txt", "w")
print("something", file=res, flush=True)

for test in tests:
    tf.random.set_seed(1234)
    init_seed(4321)

    (ds_train, ds_val, ds_test), meta = tsdf.load(
        test['dataset'],
        split=test['split'],
        data_dir=sys.path[0],
        with_info=True,
        as_supervised=True)

    num_classes = meta.features['label'].num_classes
    tmp_x = meta.features['image'].shape[0]
    tmp_y = meta.features['image'].shape[1]
    if tmp_x != None and tmp_y != None:
        test['size_x'] = tmp_x
        test['size_y'] = tmp_y
    print(test, file=res, flush=True)

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(test['size_x'], test['size_y']),
        layers.Rescaling(1./255)
    ])

    if test['augment'] == "deep":
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.Lambda(lambda x: saturate(x, 0.15)),
            layers.Lambda(lambda x: shiney(x, 0.3)),
            layers.Lambda(lambda x: contrast(x, 0.2)),
            layers.Lambda(lambda x: hue(x, 0.07)),
            layers.Lambda(lambda x: invert(x, 0.5))
            #layers.Lambda(lambda x: crop(x, 0.9, test['size_x'], test['size_y']))
        ])
    elif test['augment'] == "shallow":
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            #layers.Lambda(lambda x: shiney(x, 0.2))
        ])
    elif test['augment'] == 'none':
        data_augmentation = tf.keras.Sequential([
        ])

    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=test['learning_rate']),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    ds_train = prepare(ds_train, batch_size=test['batch_size'], shuffle=True, augment=True)
    ds_val = prepare(ds_val, batch_size=test['batch_size'])
    ds_test = prepare(ds_test, batch_size=test['batch_size'])

    #for example in ds_train:
    #    img = np.array(example[0][0])
    #    print(example)
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