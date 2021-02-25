<p align="center">
    <a href="https://github.com/ContentAutomation"><img src="https://contentautomation.s3.eu-central-1.amazonaws.com/logo.png" alt="Logo" width="150"/></a>
    <br />
    <br />
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-3C93B4.svg?style=flat" alt="MIT License"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
    <br />
    <br />
    <i>Implementation of a Neural Network that can detect whether a video is Ingame or not </i>
    <br />
<br />
    <i><b>Authors</b>:
        <a href="https://github.com/ChristianCoenen">Christian Coenen</a>,
        <a href="https://github.com/DeadlySurprise">Moritz Mundhenke</a>,
        <a href="https://github.com/lucaSchilling">Luca Schilling </a>
    </i>
</p>
<hr />


## Setup
This project requires [Poetry](https://python-poetry.org/) to install the required dependencies.
Check out [this link](https://python-poetry.org/docs/) to install Poetry on your operating system.

Make sure you have installed [Python](https://www.python.org/downloads/) 3.8 or higher! Otherwise Step 3 will let you know that you have no compatible Python version installed.

1. Clone/Download this repository
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
