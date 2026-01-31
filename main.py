import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import torch
import random
import seaborn as sns
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

# from tensorflow.keras import models, layers
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from utils.utils import *
from dataset.emnist import EMNISTDataset
from custom_network.model import CustomEMNISTCNN
from pipeline.trainer import EMNISTTrainer
from evaluation.tools import *

# EMNIST_SPLIT = 'byclass'
BATCH_SIZE = 400
VAL_RATIO = 0.1
SEED = 42

# # load labels mapping
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# filepath = os.path.join(
#     SCRIPT_DIR,
#     "mapping",
#     f"emnist-{EMNIST_SPLIT}-mapping.txt"
# )
# EMNIST_MAPPING = read_emnist_mapping(filepath)

PATIENCE = 7
N_EPOCHS = 10
LEARNING_RATE = 0.001

valid_split = ['balanced',
               'byclass',
               'bymerge',
               'digits',
               'letters',
               'mnist']

def main(args):
    EMNIST_SPLIT = args.split

    if EMNIST_SPLIT not in valid_split:
        raise ValueError(f"Unsupported mode: split {EMNIST_SPLIT}")


    print(f"[INFO] Running in {args.mode.upper()} mode")
    print(f"[INFO] Using EMNIST split: {EMNIST_SPLIT}")

    device = set_default_device()
    set_random_seed(SEED)

        # Load mapping
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "mapping", f"emnist-{EMNIST_SPLIT}-mapping.txt")
    EMNIST_MAPPING = read_emnist_mapping(filepath)
    # e.g. EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    num_classes = len(EMNIST_MAPPING)
    print(f"[INFO] Loaded {num_classes} class labels from mapping")

    # Load datasets
    full_train_dataset = EMNISTDataset('./', True, 'train', EMNIST_SPLIT)
    test_dataset = EMNISTDataset('./', True, 'test', EMNIST_SPLIT)

    dataset_size = len(full_train_dataset)

    train_indices, val_indices = train_test_split(
        list(range(dataset_size)),
        test_size=VAL_RATIO,
        stratify=full_train_dataset.targets,
        random_state=SEED
    )

    val_dataset = Subset(full_train_dataset, val_indices)
    train_dataset = Subset(full_train_dataset, train_indices)

    print(f"Dataset sizes:")
    print(f"Training: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        # worker_init_fn=worker_init_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        # worker_init_fn=worker_init_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        # worker_init_fn=worker_init_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    visualize_sample(train_dataset, 
                     'emnist_sample_visualization.png',
                     10,
                     EMNIST_SPLIT,
                     EMNIST_MAPPING)
    
    model = CustomEMNISTCNN(num_classes).to(device)

    print(f"Model initialized with {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize and run training
    train_config = {
        'epochs': N_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'patience': PATIENCE
    }
    trainer = EMNISTTrainer(model, train_loader, val_loader, test_loader, EMNIST_MAPPING, train_config=train_config)

    if args.mode == "train":
        print("[INFO] Starting training...")
        train_losses, train_accuracies, val_losses, val_accuracies = trainer.train_pipeline()
        print("[INFO] Training complete.")

        # Evaluate best model after training
        train_losses, train_accuracies, val_losses, val_accuracies = load_model('best_emnist_model_base.pth', model)
        all_predictions, all_targets, all_probabilities = evaluation(trainer, test_loader, EMNIST_MAPPING)

        # Plot training history if available
        if train_losses is not None and val_losses is not None:
            plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, "emnist_training_history.png")

        # Run inference demo
        inference(model, test_loader, device, EMNIST_MAPPING, "emnist_inference.png", n_samples=10)

    elif args.mode == "eval":
        print("[INFO] Loading model for evaluation...")

        # Evaluate best model 
        train_losses, train_accuracies, val_losses, val_accuracies = load_model('best_emnist_model_base.pth', model)
        all_predictions, all_targets, all_probabilities = evaluation(trainer, test_loader, EMNIST_MAPPING)

        # Plot training history if available
        if train_losses is not None and val_losses is not None:
            plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, "emnist_training_history.png")

        # Run inference demo
        inference(model, test_loader, device, EMNIST_MAPPING, "emnist_inference.png", n_samples=10)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate an EMNIST model")
    parser.add_argument("--mode", choices=["train", "eval", "debug"], default="train", help="Select whether to train or evaluate")
    parser.add_argument("--split", default="byclass", help="EMNIST split to use")
    # parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training/testing")
    # parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    main(args)
