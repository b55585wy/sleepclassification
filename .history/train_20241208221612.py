import os
import glob
import logging
import argparse
import itertools
from functools import reduce

import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from preprocess import preprocess
from load_files import load_npz_files
from loss_function import weighted_categorical_cross_entropy
from evaluation import f1_scores_from_cm, plot_confusion_matrix
from model import SingleSalientModel, TwoSteamSalientModel


def parse_args():
    """
    Parse command line arguments and validate them.
    """
    parser = argparse.ArgumentParser(description="Evaluate Sleep Models with K-Fold Cross-Validation.")

    parser.add_argument("--data_dir", "-d", default="./data/sleepedf-2013/npzs",
                        help="Directory where the data is stored.")
    parser.add_argument("--modal", '-m', type=int, choices=[0, 1], default=1,
                        help="Modal: 0 for single modal, 1 for multi modal.")
    parser.add_argument("--output_dir", '-o', default='./result',
                        help="Directory to save the results.")
    parser.add_argument("--valid", '-v', type=int, default=20,
                        help="Number of folds for k-fold validation.")

    args = parser.parse_args()

    # Validate k_folds
    if args.valid <= 0:
        parser.error("The `valid` argument should be a positive integer.")

    return args


def configure_logging():
    """
    Configure the logging settings.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler("evaluation.log", mode='w')
                        ])


def load_hyperparameters(hyperparams_path="hyperparameters.yaml"):
    """
    Load hyperparameters from a YAML file.
    """
    if not os.path.exists(hyperparams_path):
        logging.critical(f"Hyperparameters file '{hyperparams_path}' not found.")
        exit(-1)

    with open(hyperparams_path, encoding='utf-8') as f:
        hyper_params = yaml.safe_load(f)

    return hyper_params


def build_and_compile_model(modal, hyper_params):
    """
    Instantiate and compile the model based on the modality.
    """
    if modal == 0:
        model = SingleSalientModel(**hyper_params)
    else:
        model = TwoSteamSalientModel(**hyper_params)

    # Ensure 'class_weights' is in hyper_params for the loss function
    if 'class_weights' not in hyper_params:
        logging.critical("Missing 'class_weights' in hyperparameters.")
        exit(-1)

    loss = weighted_categorical_cross_entropy(hyper_params['class_weights'])

    model.compile(optimizer=hyper_params.get('optimizer', 'adam'),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def evaluate_model_on_fold(model, model_path, test_data, test_labels, modal, batch_size):
    """
    Load model weights and evaluate on the test data for a single fold.
    """
    # Load weights
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found.")
        return None, None, None

    model.load_weights(model_path)
    logging.info(f"Loaded weights from '{model_path}'.")

    # Predict
    if modal == 0:
        y_pred = model.predict(test_data, batch_size=batch_size)
    else:
        y_pred = model.predict(test_data, batch_size=batch_size)

    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    test_labels = test_labels.reshape((-1, test_labels.shape[-1]))

    # Compute metrics
    acc = accuracy_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    f1 = f1_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    cm = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))

    logging.info(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    logging.info("Confusion Matrix:")
    logging.info(f"{cm.astype('float32') / cm.sum():.4f}")

    return acc, f1, cm


def main():
    # Parse arguments
    args = parse_args()

    # Configure logging
    configure_logging()

    # Load hyperparameters
    hyper_params = load_hyperparameters()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load split information
    split_path = os.path.join(args.output_dir, "split.npz")
    if not os.path.exists(split_path):
        logging.critical(f"Split file '{split_path}' not found.")
        exit(-1)

    with np.load(split_path, allow_pickle=True) as f:
        npz_names = f['split']

    # Load model filenames
    model_pattern = os.path.join(args.output_dir, "fold_*_best_model.h5")
    model_names = glob.glob(model_pattern)
    model_names.sort()

    if len(model_names) < args.valid:
        logging.critical(f"Not enough models to summarize. Needed: {args.valid}, Found: {len(model_names)}.")
        exit(-1)

    logging.info(f"Found {len(model_names)} model(s) for evaluation.")

    # Build and compile the model
    model = build_and_compile_model(args.modal, hyper_params)

    # Initialize tracking variables
    best_f1 = -1.0
    best_acc = -1.0
    best_model_name = ''
    cm_list = []

    # Loop over each fold
    for i in range(args.valid):
        model_path = model_names[i]
        fold_number = i + 1
        logging.info(f"Evaluating Fold {fold_number}: '{os.path.basename(model_path)}'.")

        # Load and preprocess test data
        test_npzs = list(itertools.chain.from_iterable(npz_names[i].tolist()))
        test_data_list, test_labels_list = load_npz_files(test_npzs)

        # Convert labels to categorical
        try:
            test_labels_list = [to_categorical(label, num_classes=hyper_params.get('num_classes', 5))
                                for label in test_labels_list]
        except Exception as e:
            logging.error(f"Error converting labels to categorical: {e}")
            continue

        # Preprocess data
        try:
            test_data, test_labels = preprocess(test_data_list, test_labels_list,
                                                hyper_params.get('preprocess', {}), is_test=True)
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            continue

        logging.info(
            f"Evaluating Fold {fold_number} with {test_data[0].shape[0] if args.modal else test_data.shape[0]} samples.")

        # Evaluate model on the current fold
        acc, f1, cm = evaluate_model_on_fold(model, model_path, test_data, test_labels,
                                             args.modal, hyper_params['train']['batch_size'])

        if acc is None:
            logging.warning(f"Skipping Fold {fold_number} due to previous errors.")
            continue

        # Plot confusion matrices
        try:
            label_classes = hyper_params['evaluation']['label_class']
            plot_confusion_matrix(cm, classes=label_classes,
                                  title=f"Confusion Matrix Fold {fold_number}",
                                  path=args.output_dir,
                                  normalize=True)
            plot_confusion_matrix(cm, classes=label_classes,
                                  title=f"Confusion Matrix Numbers Fold {fold_number}",
                                  normalize=False,
                                  path=args.output_dir)
        except KeyError:
            logging.error("Missing 'label_class' in hyperparameters under 'evaluation'.")

        # Update best model tracking
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_model_name = os.path.basename(model_path)

        # Accumulate confusion matrices
        cm_list.append(cm)
        logging.info(f"Fold {fold_number} evaluation completed.\n")

        # Reset model states if necessary
        # Note: This is only needed if the model has stateful layers
        model.reset_states()

    if not cm_list:
        logging.critical("No folds were successfully evaluated.")
        exit(-1)

    # Summarize results
    logging.info(f"Best Model: {best_model_name} with Accuracy={best_acc:.4f} and F1-Score={best_f1:.4f}")

    # Sum confusion matrices
    sum_cm = reduce(lambda x, y: x + y, cm_list)
    label_classes = hyper_params['evaluation']['label_class']

    # Plot total confusion matrices
    try:
        plot_confusion_matrix(sum_cm, classes=label_classes,
                              title='Total Confusion Matrix',
                              path=args.output_dir,
                              normalize=True)
        plot_confusion_matrix(sum_cm, classes=label_classes,
                              title='Total Confusion Matrix Numbers',
                              normalize=False,
                              path=args.output_dir)
    except KeyError:
        logging.error("Missing 'label_class' in hyperparameters under 'evaluation'.")

    # Calculate average metrics
    ave_f1 = f1_scores_from_cm(sum_cm)
    ave_acc = np.trace(sum_cm) / np.sum(sum_cm)
    logging.info(f"Average Accuracy: {ave_acc:.4f}, Average F1-Score: {ave_f1:.4f}")

    print(f"Best Model: {best_model_name} with Accuracy={best_acc:.4f} and F1-Score={best_f1:.4f}")
    print(f"Average Accuracy: {ave_acc:.4f}, Average F1-Score: {ave_f1:.4f}")


if __name__ == '__main__':
    main()
