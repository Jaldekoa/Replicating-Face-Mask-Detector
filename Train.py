import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

TRAIN_PATH = "./Dataset/Train/"
VAL_PATH = "./Dataset/Validation/"
HEIGHT, WIDTH, RGB = 224, 224, 3
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4

tr_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
train = tr_data.flow_from_directory(directory = TRAIN_PATH, shuffle = True, target_size = (HEIGHT, WIDTH), class_mode = 'binary', batch_size = BATCH_SIZE)

val_data = tf.keras.preprocessing.image.ImageDataGenerator()
validation = val_data.flow_from_directory(directory = VAL_PATH, shuffle = False, target_size = (HEIGHT, WIDTH), class_mode = 'binary', batch_size = BATCH_SIZE)

baseModel = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=(HEIGHT, WIDTH, RGB)))
baseModel.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input, name='preprocessing', input_shape=(HEIGHT, WIDTH, RGB)),
    baseModel,
    tf.keras.layers.AveragePooling2D(pool_size=(7, 7)),
    tf.keras.layers.Flatten(name="flatten"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation="softmax"),
])

model.summary()

optimizer = tf.keras.optimizers.Adam(lr = LR, decay = LR / EPOCHS)
model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

model.fit(train, steps_per_epoch = train.n // BATCH_SIZE, validation_data = validation, validation_steps = validation.n // BATCH_SIZE, epochs=EPOCHS)
model.save("Detector Model.model")
