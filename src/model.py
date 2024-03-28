import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def data_augmentation(image_height: int = 224,
                      image_width: int = 224) -> tf.keras.Sequential:
    """
    Creates a data augmentation pipeline.

    Returns:
        tf.keras.Sequential: The data augmentation pipeline.
    """
    augment = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(image_height, image_width),
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.RandomContrast(0.1),
    ])
    return augment

def build_model(num_classes: int = 525) -> models.Model:
    """
    Builds a CNN model using the EfficientNetB0 architecture with pre-trained weights and custom layers.

    Args:
        num_classes (int): Number of output classes. Default is 525.

    Returns:
        models.Model: The CNN model.
    """
    # Load the pre-trained EfficientNetB0 model
    pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='max'
    )

    # Set the pre-trained model to non-trainable
    pretrained_model.trainable = False

    # Data augmentation
    augmenter = data_augmentation()

    # Custom layers on top of the pre-trained model
    inputs = pretrained_model.input
    x = augmenter(inputs)
    x = layers.Dense(128, activation='relu')(pretrained_model.output)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_callbacks() -> list:
    """
    Creates a list of callbacks for training the model.

    Returns:
        list: A list of callbacks including ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.
    """
    # Create checkpoint callback
    checkpoint_path = "birds_classification_model_checkpoint"
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          save_weights_only=True,
                                          monitor="val_accuracy",
                                          save_best_only=True)

    # Setup EarlyStopping callback
    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=5,
                                   restore_best_weights=True)

    # Setup ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=3,
                                  min_lr=1e-6)

    return [checkpoint_callback, early_stopping, reduce_lr]
