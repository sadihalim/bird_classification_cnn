import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.tensorflow

# Download and import helper functions
if not os.path.exists('helper_functions.py'):
    os.system('wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py')
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir, pred_and_plot

# Define constants
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
EPOCHS = 10

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load data
def load_data(dataset_path):
    image_dir = Path(dataset_path)
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    image_df = pd.concat([filepaths, labels], axis=1)
    return image_df

# Prepare data generators
def prepare_data_generators(train_df, test_df):
    train_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input, validation_split=0.2)
    test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    train_images = train_generator.flow_from_dataframe(dataframe=train_df, x_col='Filepath', y_col='Label', target_size=TARGET_SIZE, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=True, seed=42, subset='training')
    val_images = train_generator.flow_from_dataframe(dataframe=train_df, x_col='Filepath', y_col='Label', target_size=TARGET_SIZE, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=True, seed=42, subset='validation')
    test_images = test_generator.flow_from_dataframe(dataframe=test_df, x_col='Filepath', y_col='Label', target_size=TARGET_SIZE, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False)

    return train_images, val_images, test_images

# Build model
def build_model():
    base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(525, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(model, train_images, val_images):
    with mlflow.start_run():
        mlflow.tensorflow.autolog()

        # Create callbacks
        checkpoint_callback = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy')
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

        # Fit model
        history = model.fit(train_images, epochs=EPOCHS, validation_data=val_images, callbacks=[checkpoint_callback, early_stopping, reduce_lr, create_tensorboard_callback("training_logs", "bird_classification")])

        # Log model
        mlflow.keras.log_model(model, "models")

    return history

# Load model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

if __name__ == '__main__':
    dataset_path = "../input/100-bird-species/train"
    image_df = load_data(dataset_path)
    train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=
