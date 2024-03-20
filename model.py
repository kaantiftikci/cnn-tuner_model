import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    zoom_range=0.15,  
    shear_range=0.15  
)
x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

def build_model(hp):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(hp.Int('conv1_units', min_value=32, max_value=256, step=32), (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)),

        tf.keras.layers.Conv2D(hp.Int('conv2_units', min_value=32, max_value=256, step=32), (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int('dense_units', min_value=256, max_value=1024, step=256), activation='relu'),  # Daha büyük birimler kullan
        tf.keras.layers.Dropout(hp.Float('dropout_3', 0, 0.5, step=0.1)),
        tf.keras.layers.Dense(100, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt_updated')


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8) 

tuner.search(datagen.flow(x_train_new, y_train_new, batch_size=64),
             epochs=20,
             validation_data=(x_val, y_val),
             callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(
    datagen.flow(x_train_new, y_train_new, batch_size=64),
    epochs=50,  # Daha fazla epoch
    validation_data=(x_val, y_val),
    callbacks=[stop_early],
    verbose=1
)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
