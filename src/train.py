import mlflow
import mlflow.keras
import os
from data_loader import load_data
from model import build_model, get_callbacks

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
    train_data, val_data, test_data = load_data(train_dir, test_dir)

    # Build the model
    model = build_model(num_classes=525)

    # Define training parameters
    epochs = 10
    batch_size = 32

    # Start an MLflow run
    with mlflow.start_run():
        # Train the model
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=get_callbacks(),
            verbose=1
        )

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_data)

        # Log parameters, metrics, and the model
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size})
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
        mlflow.keras.log_model(model, "model")

        # Save the trained model for later inference
        model_save_path = os.path.join('models', 'bird_classification_model.h5')
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
