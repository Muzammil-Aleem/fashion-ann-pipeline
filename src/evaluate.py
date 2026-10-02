import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model


def main():
    # Load trained model
    model = load_model("models/model.h5")

    # Load processed test data
    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    # Generate predictions
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap="Blues")
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Evaluation completed successfully.")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Metrics saved to metrics.json")
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()