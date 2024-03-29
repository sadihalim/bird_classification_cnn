import mlflow
import mlflow.keras
import os
from data_loader import load_data, prepare_dataframes
from model import build_model, get_callbacks
from tensorflow.keras.optimizers import Adam

def train() -> None:
    """
    Trains the bird classification model, tracks the training process using MLflow,
    and saves the model for later inference.

    The function performs the following steps:
    1. Set up MLflow tracking.
    2. Load the training, validation, and test data.
    3. Build the model.
    4. Define training parameters.
    5. Train the model and log the training process using MLflow.
    6. Evaluate the model on the test set and log the results.
    7. Save the trained model for later inference.
    """
    # Set up MLflow tracking
    mlflow.set_experiment("Bird_Classification")

    # Load the data
    train_dir = os.path.join('data', 'train')
    test_dir = os.path.join('data', 'test')
    train_data, test_data = prepare_dataframes(train_dir)

    train_df, test_df = load_data(train_data,test_data)
    # Build the model
    model = build_model(num_classes=525)
    optimizer = Adam(0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define training parameters
    epochs = 150
    batch_size = 32

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "optimizer": type(optimizer).__name__})

        
        # Train the model
        history = model.fit(
            train_df,
            epochs=epochs,
            callbacks=get_callbacks(),  # assuming get_callbacks returns the required callbacks
            verbose=1
        )

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_df)

        # Log metrics
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})

        # Log the model
        mlflow.keras.log_model(model, "model")

        # Save the trained model for later inference
        model_save_path = os.path.join('models', 'bird_classification_model.h5')
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


train()
