from typing import Tuple
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from helper_functions import walk_through_dir


def prepare_dataframes(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares dataframes for the training and validation datasets by walking through each directory.

    Args:
        dataset (str): Path to the dataset directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframes for the training and validation datasets.
    """

    walk_through_dir(dataset)
    image_dir = Path(dataset)

    # Get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)

    # Split into train and test dataframes
    train_df, test_df = train_test_split(image_df,
                                         test_size=0.2,
                                         shuffle=True,
                                         random_state=42)

    return train_df, test_df


def load_data(train_df: pd.DataFrame, test_df: pd.DataFrame, img_height: int = 224, img_width: int = 224, batch_size: int = 32) -> Tuple[DirectoryIterator, DirectoryIterator]:
    """
    Loads and preprocesses the dataset from dataframes.

    Args:
        train_df (pd.DataFrame): DataFrame containing filepaths and labels for the training data.
        test_df (pd.DataFrame): DataFrame containing filepaths and labels for the test data.
        img_height (int): Height of the input images. Default is 224.
        img_width (int): Width of the input images. Default is 224.
        batch_size (int): Size of the batches of data. Default is 32.

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator]: Training and test data generators.
    """

    # Training data generator
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=0.2
        )
    train_data = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Test data generator
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    )
    test_data = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, test_data
