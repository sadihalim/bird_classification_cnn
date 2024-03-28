import os
from tensorflow.keras.models import load_model
from data_loader import load_data

def evaluate(model_path: str,
             test_dir: str,
             img_height: int = 224,
             img_width: int = 224,
             batch_size: int = 32) -> None:
    """
    Evaluates the trained model on the test dataset.

    Args:
        model_path (str): Path to the saved model.
        test_dir (str): Path to the test directory.
        img_height (int): Height of the input images. Default is 224.
        img_width (int): Width of the input images. Default is 224.
        batch_size (int): Size of the batches of data. Default is 32.
    """
    # Load the saved model
    model = load_model(model_path)

    # Load the test data
    _, _, test_data = load_data(train_dir='',
                                test_dir=test_dir,
                                validation_split=0,
                                img_height=img_height,
                                img_width=img_width,
                                batch_size=batch_size)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if __name__ == '__main__':
    evaluate(model_path='models/bird_classification_model.h5', test_dir=os.path.join('data', 'test'))
