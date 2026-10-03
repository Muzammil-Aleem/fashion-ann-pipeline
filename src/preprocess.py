import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    params = load_params()

    validation_size = params["preprocess"]["validation_size"]
    random_seed = params["preprocess"]["random_seed"]

    os.makedirs("data/processed", exist_ok=True)

    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    x_train = (x_train.astype("float32") / 255.0) * 0.9
    x_test = (x_test.astype("float32") / 255.0) * 0.9
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_size,
        random_state=random_seed,
        stratify=y_train
    )

    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)

    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)

    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing completed successfully.")
    print(f"Training set: {x_train.shape}")
    print(f"Validation set: {x_val.shape}")
    print(f"Test set: {x_test.shape}")


if __name__ == "__main__":
    main()