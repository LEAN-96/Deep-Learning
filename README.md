# Deep Learning - TensorFlow Notebook

## 1. Project Overview

This project demonstrates the implementation of a machine learning pipeline using TensorFlow. The primary objective is to train a model on a dataset, evaluate its performance, and visualize the results. The notebook covers data preprocessing, model building, training, and evaluation using TensorFlow's deep learning framework.

### Dataset
The project utilizes a dataset suitable for classification tasks (e.g., MNIST or CIFAR-10). The data is preprocessed and split into training and testing sets. The notebook walks through the steps of loading the dataset, normalizing it, and preparing it for model training.

### Machine Learning Methods
The notebook implements a convolutional neural network (CNN) using TensorFlow and Keras. The model is trained on the dataset, and its performance is evaluated using metrics such as accuracy and loss.


### Notebook Overview
The notebook contains several sections:
1. Data Loading and Preprocessing:
2. Loads the dataset and preprocesses it for training.
3. **Model Building**: Defines a convolutional neural network (CNN) using TensorFlow's Keras API.
4. **Model Training**: Trains the CNN on the training data.
5. **Evaluation**: Evaluates the model on test data and displays metrics like accuracy and loss.
6. **Visualization**: Plots graphs to visualize model performance over epochs.
---

## 2. Prerequisites and Installation

### Prerequisites
To run the notebook locally or online via MyBinder, you need the following:
- **Python 3.8+**
- **Jupyter Notebook** or **JupyterLab**
- **TensorFlow 2.x**
- Other Python libraries listed in `environment.yml`

### Dependencies
The project relies on several Python packages. You can find all dependencies listed in the `environment.yml` file. Some key dependencies include:
- `tensorflow`
- `numpy`
- `matplotlib`
- `pandas`


### Running Locally

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LEAN-96/Deep-Learning.git
    cd decision-trees-random-forests
    ```

2. **Create and activate the environment**:
    Make sure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
    ```bash
    conda env create -f environment.yml
    conda activate decision-tree-random-forest-environment
    ```

3. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
    Open `5-Tensorflow.ipynb` in the Jupyter interface to run the notebook.


### Running Online via MyBinder
What is MyBinder?
MyBinder is a free service that allows you to run Jupyter notebooks online without needing to install anything locally. It creates an interactive environment where you can execute notebooks directly from your browser.
Running this Notebook on MyBinder
Click on this link to launch MyBinder with this repository:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LEAN-96/Deep-Learning.git/HEAD?labpath=notebooks)

Once MyBinder loads, navigate to the notebook (5-Tensorflow_Projekt.ipynb) in the file browser on the left side of your screen.
Open the notebook by clicking on it.
To run all cells in sequence, click "Cell" -> "Run All" from the top menu or run each cell individually by selecting it and pressing Shift + Enter.
Wait for each cell to execute; outputs will appear below each code block as they complete.

## 3. Reproducing the Results

1. Open the provided notebook (5-Tensorflow_Projekt.ipynb) from the Jupyter interface.
2. Run each cell sequentially by pressing Shift + Enter or by clicking "Run" in the Jupyter toolbar.
3. Ensure that all cells execute without errors.
4. At the end of execution, observe accuracy metrics and loss plots to evaluate model performance.
5. **Interpreting Results**:
- Accuracy: The accuracy metric shows how well the model performs on both training and test datasets.
- Loss: Loss values indicate how well the model fits the data; lower values are better.
- Plots: Visualizations of accuracy and loss over epochs help to understand how well the model is learning during training.


