# Check GPU availability
#!nvidia-smi

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import TFAutoModelForImageClassification
from itertools import product
import tensorflow as tf

# Parameters
IMAGE_SIZE = (224, 224)  # Default image size for resizing
BATCH_SIZE = 32  # Default batch size
DEFAULT_EPOCHS = 10
EXPERIMENT_RESULTS = []  # To store results of all experiments

# Data Preprocessing

def create_image_generators(base_path, img_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Creates data generators for training, validation, and testing.
    Filters dataset to include only .jpeg or .jpg files.

    Args:
        base_path (str): Path to the dataset directory.
        img_size (tuple): Target size for resizing images.
        batch_size (int): Batch size for the generators.

    Returns:
        tuple: Training, validation, and test generators.
    """
    def jpeg_filter(file_path):
        return file_path.endswith(".jpeg") or file_path.endswith(".jpg")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        preprocessing_function=lambda x: x if jpeg_filter(x) else None
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        preprocessing_function=lambda x: x if jpeg_filter(x) else None
    )

    train_gen = train_datagen.flow_from_directory(
        base_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        base_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_gen, val_gen, test_datagen

# Model Setup

def build_model(model_type, img_size, num_classes, learning_rate, optimizer):
    """
    Builds a CNN or transformer-based model based on the provided parameters.

    Args:
        model_type (str): Model type (e.g., MobileNet, ResNet50, ViT, SWIN).
        img_size (tuple): Input image size.
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
        optimizer (str): Optimizer type (Adam, SGD).

    Returns:
        keras.Model: Compiled model.
    """
    if model_type in ["MobileNet", "ResNet50"]:
        base_model = MobileNet if model_type == "MobileNet" else ResNet50
        model = Sequential([
            base_model(include_top=False, input_shape=img_size + (3,), pooling="avg", weights="imagenet"),
            Dense(num_classes, activation="softmax")
        ])
    else:
        model = TFAutoModelForImageClassification.from_pretrained(
            f"{model_type.lower()}-base-patch16-224-in21k",
            num_labels=num_classes
        )

    optimizer_instance = Adam(learning_rate=learning_rate) if optimizer == "Adam" else SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer_instance, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Training

def train_model(model, train_gen, val_gen, epochs):
    """
    Trains the model.

    Args:
        model (keras.Model): Compiled model.
        train_gen: Training data generator.
        val_gen: Validation data generator.
        epochs (int): Number of epochs.

    Returns:
        keras.callbacks.History: Training history.
    """
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    checkpoint_path = "best_model.h5"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor="val_loss", mode="min")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )
    print(f"Model checkpoint saved at {checkpoint_path}")
    return history

# Visualization

def plot_results(history, title):
    """
    Plots training and validation accuracy.

    Args:
        history (keras.callbacks.History): Training history.
        title (str): Plot title.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plot_path = f"{title.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved: {plot_path}")

# Evaluation

def evaluate_model(model, val_data, class_labels, model_type, optimizer, img_size):
    """
    Evaluates the model and saves results.

    Args:
        model (keras.Model): Trained model.
        val_data: Validation data generator.
        class_labels (list): List of class labels.
        model_type (str): Model type.
        optimizer (str): Optimizer type.
        img_size (int): Image size.

    Returns:
        dict: Evaluation results.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    predictions = model.predict(val_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_data.classes[val_data.index_array]

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    accuracy = report['accuracy']

    # Save classification report
    report_path = f"classification_report_{model_type}_{optimizer}_{img_size}_{timestamp}.txt"
    with open(report_path, "w") as file:
        file.write(classification_report(y_true, y_pred, target_names=class_labels))
    print(f"Classification report saved: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = f"confusion_matrix_{model_type}_{optimizer}_{img_size}_{timestamp}.png"
    plt.savefig(cm_path)
    plt.show()
    print(f"Confusion matrix saved: {cm_path}")

    return {
        'model_type': model_type,
        'optimizer': optimizer,
        'img_size': img_size,
        'accuracy': accuracy,
        'classification_report': report_path,
        'confusion_matrix': cm_path
    }

# K-Fold Cross-Validation

def k_fold_cross_validation(dataset_path, model_type, img_size, batch_size, learning_rate, epochs, optimizer, k=5):
    """
    Performs k-fold cross-validation.

    Args:
        dataset_path (str): Path to the dataset.
        model_type (str): Type of model to use.
        img_size (tuple): Image size for resizing.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        epochs (int): Number of epochs.
        optimizer (str): Optimizer type.
        k (int): Number of folds.

    Returns:
        None
    """
    all_data = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    X, y = all_data.filenames, all_data.classes

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Running Fold {fold}/{k}")
        train_data = [X[i] for i in train_idx]
        val_data = [X[i] for i in val_idx]

        train_gen = all_data
        val_gen = all_data

        num_classes = len(all_data.class_indices)
        class_labels = list(all_data.class_indices.keys())

        model = build_model(model_type, img_size, num_classes, learning_rate, optimizer)

        history = train_model(model, train_gen, val_gen, epochs)
        plot_results(history, f"Fold {fold}: {model_type} with {optimizer} Optimizer at {img_size[0]}px")

        results = evaluate_model(model, val_gen, class_labels, model_type, optimizer, img_size[0])
        results['fold'] = fold
        EXPERIMENT_RESULTS.append(results)

# Auto-Generated Report

def generate_report():
    """
    Generates a summary report of all experiments.

    Returns:
        None
    """
    results_df = pd.DataFrame(EXPERIMENT_RESULTS)
    report_path = "summary_report.html"
    results_df.to_html(report_path, index=False)
    print(f"Summary report saved to {report_path}")



def main(dataset_path, model_type, img_size, batch_size, learning_rate, epochs, optimizer):
    """
    Main function to run the pipeline for training and evaluation.

    Args:
        dataset_path (str): Path to the dataset directory.
        model_type (str): Type of model to train (e.g., "MobileNet", "ResNet50").
        img_size (int): Image size for resizing (e.g., 224 for 224x224).
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        optimizer (str): Optimizer type (e.g., "Adam", "SGD").

    Returns:
        dict: Results of the experiment.
    """
    # Load data generators
    train_gen, val_gen, _ = create_image_generators(dataset_path, img_size=(img_size, img_size), batch_size=batch_size)

    # Get number of classes and class labels
    num_classes = len(train_gen.class_indices)
    class_labels = list(train_gen.class_indices.keys())

    # Build and train the model
    model = build_model(model_type, (img_size, img_size), num_classes, learning_rate, optimizer)
    history = train_model(model, train_gen, val_gen, epochs)

    # Plot training results
    plot_results(history, f"{model_type} with {optimizer} Optimizer at {img_size}px")

    # Evaluate the model
    results = evaluate_model(model, val_gen, class_labels, model_type, optimizer, img_size)
    EXPERIMENT_RESULTS.append(results)
    return results




# Example Grid Search
def grid_search():
    params_grid = {
        "dataset_path": ["dataset/Xray", "dataset/CT-Scan"],
        "model_type": ["MobileNet", "ResNet50", "ViT", "SWIN"],
        "img_size": [224, 256],
        "batch_size": [32, 64],
        "learning_rate": [0.001, 0.0001],
        "epochs": [10, 50],
        "optimizer": ["Adam", "SGD"]
    }

    for params in product(*params_grid.values()):
        param_dict = dict(zip(params_grid.keys(), params))
        print(f"Running experiment with parameters: {param_dict}")
        main(**param_dict)

    # Save results as a DataFrame
    results_df = pd.DataFrame(EXPERIMENT_RESULTS)
    results_csv_path = "experiment_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"All experiment results saved to {results_csv_path}")
    generate_report()






if __name__ == "__main__":
    grid_search()
