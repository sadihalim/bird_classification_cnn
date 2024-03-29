import os
from src.data_loader import prepare_dataframes, load_data
from src.model import build_model, get_callbacks
from src.train import train
from src.evaluate import evaluate


def main():
    """
    Main function to prepare dataframes, load the data, train the model, and evaluate it on the test set.
    """
    # Set paths
    train_dir = os.path.join('/data', 'train')
    test_dir = os.path.join('/data', 'test')
    print(test_dir)
    model_path = 'models/bird_classification_model.h5'

    # Prepare dataframes for the training and validation datasets
    train_df, test_df = prepare_dataframes(train_dir)

    # Load and preprocess the data
    train_data, test_data = load_data(train_df, test_df)

    # Build and train the model
    model = build_model(num_classes=525)
    train(model, train_data, test_data)

    # Save the trained model
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Evaluate the model on the test set
    evaluate(model_path, test_dir)

if __name__ == '__main__':
    main()
