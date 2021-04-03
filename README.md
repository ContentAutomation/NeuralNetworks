<p align="center">
    <a href="https://github.com/ContentAutomation"><img src="https://contentautomation.s3.eu-central-1.amazonaws.com/logo.png" alt="Logo" width="150"/></a>
    <br />
    <br />
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-3C93B4.svg?style=flat" alt="MIT License"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
    <br />
    <a href="https://github.com/tensorflow/tensorflow"><img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?logo=tensorflow&logoColor=white" alt="Tensorflow"></a>
    <br />
    <br />
    <i>Implementation of a Neural Network that can detect whether a video is Ingame or not </i>
    <br />
<br />
    <i><b>Authors</b>:
        <a href="https://github.com/ChristianCoenen">Christian C.</a>,
        <a href="https://github.com/MorMund">Moritz M.</a>,
        <a href="https://github.com/lucaSchilling">Luca S.</a>
    </i>
    <br />
    <i><b>Related Projects</b>:
        <a href="https://github.com/ContentAutomation/TwitchCompilationCreator">Twitch Compilation Creator</a>,
        <a href="https://github.com/ContentAutomation/YouTubeUploader">YouTube Uploader</a>,
        <a href="https://github.com/ContentAutomation/YouTubeWatcher">YouTube Watcher</a>
    </i>
</p>
<hr />

<p align="center">
    <img src="https://contentautomation.s3.eu-central-1.amazonaws.com/detect_ingame.png" alt="Ingame Detection" width="1200"/></a>
</p>

## About
This project implements a convolutional neural network architecture that can be trained to detect whether a given video clip is in-game or not. The network is trained using transfer learning by choosing one of the following architectures: ResNet50 (default), VGG16, InceptionV3

## Setup
This project requires [Poetry](https://python-poetry.org/) to install the required dependencies.
Check out [this link](https://python-poetry.org/docs/) to install Poetry on your operating system.

Make sure you have installed [Python](https://www.python.org/downloads/) 3.8 or higher! Otherwise Step 3 will let you know that you have no compatible Python version installed.

1. Clone/Download this repository
   
   *Note: For test models/assets, download [Release v1.0](https://github.com/ContentAutomation/NeuralNetworks/releases/latest)*
2. Navigate to the root of the repository
3. Run ```poetry install``` to create a virtual environment with Poetry
4. Run ```poetry run python src/filename.py``` to run the program. Alternatively you can run ```poetry shell``` followed by ```python src/filename.py```
5. Enjoy :)

## Script Explanations

### video2images.py
This utility can be used to build the dataset by splitting video files into images.

### predict.py
This script is used to verify the performance of the trained neural network 
by specifying a path to the model of the trained neural network,
and a video clip that should be analyzed.

### game_detection.py
This script is used to train a neural network (e.g. create a model) on a given dataset.
If enough data is present, the neural network will learn to distinguish Ingame clips from clips
that are not ingame (e.g. Lobby, Queue, ...)

## Creating a new model for a game
Let's assume you want to create a new model for the game Dota2. The following steps have to be performed:
1. Download clips for Dota2 that are both ingame and not ingame (recommended source: Twitch)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; HINT: You can download clips manually or by creating a compilation with [TwitchCompilationCreator](https://github.com/ContentAutomation/TwitchCompilationCreator) 

2. Split the clips into images via ```video2images.py```
3. Create the following folder structure
```bash
...
│
└───anyFolderName
    │
    └───dota2
    └───nogame
```
4. Sort the clips from step 1 into those folders depending on if they are ingame or not
5. Create a ```main.py``` file in ```./src/``` to initialize a ```GameDetection``` object, then run it (see example below) 
6. Test the created model on a few example clips using ```predict.py``` to verify its accuracy

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; NOTE: The number of images in the 'gamename' or 'nogame' folder has to be greater than or equal to the defined batch size
```python
# For more information about the parameters, check out game_detection.py
m = GameDetection(
    model_name="ResNet50",
    game_name="dota2",
    dataset_path="---PATH TO 'anyFolderName'---",
    input_size=(224, 224),
    batch_size=16,
    save_generated_images=False,
    convert_to_gray=False,
)
m.train(epochs=2)
```
