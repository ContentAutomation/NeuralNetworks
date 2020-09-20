import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from pathlib import Path
import numpy as np

class GameDetection:

    def __init__(self,
                 game_name,
                 dataset_path,
                 input_size,
                 batch_size,
                 save_generated_images=False,
                 convert_to_gray=False
                 ):
        self.batch_size = batch_size
        self.game_name = game_name
        self.convert_to_gray = convert_to_gray
        self.GENERATOR_IMAGES_FOLDER_NAME = 'gen'
        Path(self.GENERATOR_IMAGES_FOLDER_NAME).mkdir(parents=True, exist_ok=True) if save_generated_images else None

        self.data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.2,
            preprocessing_function=self.preprocessor
        )

        self.train_generator = self.data_gen.flow_from_directory(
            directory=dataset_path,
            target_size=input_size,
            color_mode="rgb",
            classes=[game_name, 'nogame'],
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=None,
            save_to_dir=self.GENERATOR_IMAGES_FOLDER_NAME if save_generated_images else None,
            save_prefix="",
            save_format="png",
            follow_links=False,
            subset='training',
            interpolation="nearest",
        )

        self.validation_generator = self.data_gen.flow_from_directory(
            directory=dataset_path,
            target_size=input_size,
            color_mode="rgb",
            classes=[game_name, 'nogame'],
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=None,
            save_to_dir=None,
            save_prefix="",
            save_format="png",
            follow_links=False,
            subset='validation',
            interpolation="nearest",
        )

        self.model = self.create_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def create_model(self):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
        # base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        # base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        return model

    def train(self, epochs):
        # train the model
        self.model.fit(self.train_generator,
                       steps_per_epoch=self.train_generator.samples // self.batch_size,
                       validation_data=self.validation_generator,
                       validation_steps=self.validation_generator.samples // self.batch_size,
                       epochs=epochs)

        self.model.save(f"game_detection_{self.game_name}.h5")

    def preprocessor(self, image):
        if self.convert_to_gray:
            image = rgb2gray(image)
            image = np.repeat(image[..., np.newaxis], 3, -1)
        return image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def preprocessor(image, convert_to_gray):
    if convert_to_gray:
        image = rgb2gray(image)
        image = np.repeat(image[..., np.newaxis], 3, -1)
    return image


m = GameDetection(game_name='leagueoflegends',
                  dataset_path='/Users/christiancoenen/Google Drive/Social Media Automation/datasets/gameDetection',
                  input_size=(224, 224),
                  batch_size=16,
                  save_generated_images=False,
                  convert_to_gray=False
                  )
m.train(epochs=2)
