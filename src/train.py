import os
import csv
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    params = load_params()
    train_params = params["train"]

    dense_units = train_params["dense_units"]
    dropout_rate = train_params["dropout_rate"]
    learning_rate = train_params["learning_rate"]
    epochs = train_params["epochs"]
    batch_size = train_params["batch_size"]

    os.makedirs("models", exist_ok=True)

    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    model.save("models/model.h5")

    with open("models/history.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])

        for i in range(len(history.history["loss"])):
            writer.writerow([
                i + 1,
                history.history["loss"][i],
                history.history["accuracy"][i],
                history.history["val_loss"][i],
                history.history["val_accuracy"][i]
            ])

    print("Model training completed successfully.")
    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")


if __name__ == "__main__":
    main()