import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def main():
    os.makedirs("data/raw", exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)

    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

    print("Fashion-MNIST data prepared successfully.")
    print(f"Training images shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test images shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")


if __name__ == "__main__":
    main()