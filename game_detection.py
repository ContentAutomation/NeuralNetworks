import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, LeakyReLU

IMG_PATH = '/Users/christiancoenen/datasets/gameDetection'
BATCH_SIZE = 16
INPUT_SIZE = (224, 224)
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

train_generator = data_gen.flow_from_directory(
    directory=IMG_PATH,
    target_size=INPUT_SIZE,
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    follow_links=False,
    subset='training',
    interpolation="nearest",
)
validation_generator = data_gen.flow_from_directory(
    directory=IMG_PATH,
    target_size=INPUT_SIZE,
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    follow_links=False,
    subset='validation',
    interpolation="nearest",
)

base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
# base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
# base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation=LeakyReLU(alpha=0.2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation=LeakyReLU(alpha=0.2))(x)
x = tf.keras.layers.BatchNormalization()(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input,  outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // BATCH_SIZE,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // BATCH_SIZE,
          epochs=5)

# model.save(".")
