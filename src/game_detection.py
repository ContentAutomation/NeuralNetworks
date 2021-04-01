from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU


class GameDetection:
    def __init__(
        self,
        model_name,
        game_name,
        dataset_path,
        input_size,
        batch_size,
        save_generated_images=False,
        convert_to_gray=False,
    ):
        """
        Classification Network that determines whether clips are ingame or not.

        :param game_name: The name of the game that the model is trained on.
                          The 'dataset_path' must contain a subfolder with the 'game_name' containing the ingame images!
        :param dataset_path: Path to the folder that contains two sub folders containing the images.
                             1st sub folder must have the same name as the given 'game_name' (contains ingame images)
                             2nd sub folder must be named 'nogame' and should contain the images that are NOT ingame
        :param input_size: The size to which the dataset images are scaled to (Recommended: 224, 224)
        :param batch_size: Determines how many images are used per backpropagation step.
        :param save_generated_images: Whether to save images that are fed into the neural network.
                                      This might be useful when applying data augmentation!
        :param convert_to_gray: Whether the images are converted to grayscale before feeding into the neural network!
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.game_name = game_name
        self.convert_to_gray = convert_to_gray
        self.GENERATOR_IMAGES_FOLDER_NAME = "gen"
        Path(self.GENERATOR_IMAGES_FOLDER_NAME).mkdir(parents=True, exist_ok=True) if save_generated_images else None

        self.data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.2, preprocessing_function=self.preprocessor
        )

        self.train_generator = self.data_gen.flow_from_directory(
            directory=dataset_path,
            target_size=input_size,
            color_mode="rgb",
            classes=[game_name, "nogame"],
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=None,
            save_to_dir=self.GENERATOR_IMAGES_FOLDER_NAME if save_generated_images else None,
            save_prefix="",
            save_format="png",
            follow_links=False,
            subset="training",
            interpolation="nearest",
        )

        self.validation_generator = self.data_gen.flow_from_directory(
            directory=dataset_path,
            target_size=input_size,
            color_mode="rgb",
            classes=[game_name, "nogame"],
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=None,
            save_to_dir=None,
            save_prefix="",
            save_format="png",
            follow_links=False,
            subset="validation",
            interpolation="nearest",
        )

        self.model = self.create_model()
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def create_model(self):
        if self.model_name == "ResNet50":
            base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
        elif self.model_name == "VGG16":
            base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        elif self.model_name == "InceptionV3":
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
        else:
            raise ValueError("Invalid model name! Please choose from: 'ResNet50, VGG16, InceptionV3'")

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
        predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        return model

    def train(self, epochs):
        # train the model
        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.batch_size,
            epochs=epochs,
        )

        self.model.save(f"game_detection_{self.game_name}.h5")

    def preprocessor(self, image):
        if self.convert_to_gray:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            image = np.repeat(image[..., np.newaxis], 3, -1)
        return image


"""
Example code to train a model.
Please fill 'game_name' and 'dataset_path' accordingly before running it!
"""
m = GameDetection(
    model_name="ResNet50",
    game_name="---REPLACE---",
    dataset_path="---REPLACE---",
    input_size=(224, 224),
    batch_size=16,
    save_generated_images=False,
    convert_to_gray=False,
)
m.train(epochs=2)
